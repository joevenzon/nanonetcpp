#pragma once

#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "autograd.h"

#define CHECKPOINT_MAGIC "NNCHK001"
#define CHECKPOINT_VERSION 2

template <typename DataType>
struct ParameterCheckpoint
{
	// the model weights themselves
	std::vector <DataType> values;

	// annotations about what the weights mean
	std::vector <typename AutoGrad<DataType>::LeafParameterRecord> leaf_params;

	// for gradient accumulation
	std::vector<DataType> grads;

	// for the optimizer
	std::vector<DataType> moment1;		// Adam first moment
	std::vector<DataType> moment2;		// Adam second moment
	int step_count = 0;					// for Adam bias correction

	void init(AutoGrad<DataType> & grad)
	{
		values.resize(grad.get_values().size(), 0);
		grads.resize(grad.get_gradients().size(), 0);
		moment1.resize(values.size(), 0);
		moment2.resize(values.size(), 0);

		for (int i = 0; i < values.size(); i++)
		{
			values[i] = grad.get_values()[i];
		}

		step_count = 0;

		update(grad);
	}

	// pass accumulate=true during a gradient accumulation cycle
	void update(AutoGrad<DataType> & grad, bool accumulate = false)
	{
		for (int i = 0; i < values.size(); i++)
		{
			values[i] = grad.get_values()[i];
		}

		if (accumulate)
		{
			for (int i = 0; i < values.size(); i++)
			{
				grads[i] += grad.get_gradients()[i];
			}
		}
		else
		{
			for (int i = 0; i < values.size(); i++)
			{
				grads[i] = grad.get_gradients()[i];
			}
		}

		leaf_params.assign(grad.get_leaf_parameters().begin(), grad.get_leaf_parameters().end());
	}

	// Average the accumulated gradients over the number of micro-steps, for gradient accumulation.
	void scale_grads(DataType scale)
	{
		for (DataType & g : grads)
			g *= scale;
	}

	size_t size() const { return values.size(); }

	/// Save checkpoint to an ostream in binary format.
	///
	/// Format:
	///   Header (32 bytes)
	///     - Magic:            8 bytes  "NNCHK001" or similar
	///     - Version:          4 bytes  uint32_t (little-endian)
	///     - DataTypeID:       4 bytes  uint32_t (1 = 32-bit float)
	///     - ParamCount:       8 bytes  uint64_t (number of parameters)
	///     - StepCount:        4 bytes  int (signed, for Adam bias correction)
	///     - LeafParamCount:   4 bytes  uint32_t (number of LeafParameterRecord entries)
	///   Leaf Parameter Records (variable)
	///     For each record:
	///       - params.start:   4 bytes  int (NodeHandle)
	///       - params.rows:    4 bytes  int
	///       - params.cols:    4 bytes  int
	///       - name.length:    4 bytes  uint32_t
	///       - name.data:      variable bytes (UTF-8, no null terminator)
	///   Payload (variable)
	///     - values[]   paramCount * sizeof(DataType)
	///     - grads[]    paramCount * sizeof(DataType)
	///     - moment1[]  paramCount * sizeof(DataType)
	///     - moment2[]  paramCount * sizeof(DataType)
	bool save(std::ostream &out, std::ostream &err) const
	{
		if (values.empty())
		{
			err << "Cannot save an empty checkpoint\n";
			return false;
		}

		// --- header ---
		constexpr char magic[9] = CHECKPOINT_MAGIC;
		out.write(magic, 8);

		uint32_t version = CHECKPOINT_VERSION;
		out.write(reinterpret_cast<const char *>(&version), sizeof(version));

		uint32_t dataTypeId = 0;
		if constexpr (std::is_same_v<DataType, float>)
			dataTypeId = 1;
		else
		{
			err << "Unsupported DataType for checkpoint save\n";
			return false;
		}
		out.write(reinterpret_cast<const char *>(&dataTypeId), sizeof(dataTypeId));

		uint64_t paramCount = static_cast<uint64_t>(values.size());
		out.write(reinterpret_cast<const char *>(&paramCount), sizeof(paramCount));

		out.write(reinterpret_cast<const char *>(&step_count), sizeof(step_count));

		// leaf params count
		uint32_t leafCount = static_cast<uint32_t>(leaf_params.size());
		out.write(reinterpret_cast<const char *>(&leafCount), sizeof(leafCount));

		if (!out)
		{
			err << "I/O error while writing checkpoint header";
			return false;
		}

		// --- leaf param records ---
		for (auto& rec : leaf_params)
		{
			out.write(reinterpret_cast<const char *>(&rec.params.start), sizeof(rec.params.start));
			out.write(reinterpret_cast<const char *>(&rec.params.rows), sizeof(rec.params.rows));
			out.write(reinterpret_cast<const char *>(&rec.params.cols), sizeof(rec.params.cols));

			uint32_t nameLen = static_cast<uint32_t>(rec.name.size());
			out.write(reinterpret_cast<const char *>(&nameLen), sizeof(nameLen));
			if (nameLen > 0)
			{
				out.write(rec.name.c_str(), static_cast<std::streamsize>(nameLen));
			}
		}

		if (!out)
		{
			err << "I/O error while writing leaf parameter records";
			return false;
		}

		// --- payload ---
		out.write(reinterpret_cast<const char *>(values.data()),
		          static_cast<std::streamsize>(paramCount * sizeof(DataType)));
		out.write(reinterpret_cast<const char *>(grads.data()),
		          static_cast<std::streamsize>(paramCount * sizeof(DataType)));
		out.write(reinterpret_cast<const char *>(moment1.data()),
		          static_cast<std::streamsize>(paramCount * sizeof(DataType)));
		out.write(reinterpret_cast<const char *>(moment2.data()),
		          static_cast<std::streamsize>(paramCount * sizeof(DataType)));

		if (!out)
		{
			err << "I/O error while writing checkpoint data";
			return false;
		}

		return true;
	}

	/// Load checkpoint from an istream in binary format.
	///
	/// @throws std::runtime_error on I/O errors, bad magic, unsupported version,
	///         or DataType mismatch.
	bool load(std::istream &in, std::ostream & err)
	{
		// --- header ---
		char magic[9] = {};
		in.read(magic, 8);
		if (std::strcmp(magic, CHECKPOINT_MAGIC) != 0)
		{
			err << "Invalid checkpoint file: bad magic\n";
			return false;
		}

		uint32_t version = 0;
		in.read(reinterpret_cast<char *>(&version), sizeof(version));
		if (version != CHECKPOINT_VERSION)
		{
			err << "Unsupported checkpoint version: " << version << "\n";
			return false;
		}

		uint32_t dataTypeId = 0;
		in.read(reinterpret_cast<char *>(&dataTypeId), sizeof(dataTypeId));
		bool typeMatch = false;
		if constexpr (std::is_same_v<DataType, float>)
			typeMatch = (dataTypeId == 1);
		if (!typeMatch)
		{
			err << "DataType mismatch: file stores id=" << dataTypeId << " which is unsupported\n";
			return false;
		}

		uint64_t paramCount = 0;
		in.read(reinterpret_cast<char *>(&paramCount), sizeof(paramCount));

		int loadedStep = 0;
		in.read(reinterpret_cast<char *>(&loadedStep), sizeof(loadedStep));

		// leaf parameter count
		uint32_t leafCount = 0;
		in.read(reinterpret_cast<char *>(&leafCount), sizeof(leafCount));

		if (!in)
		{
			err << "I/O error while reading checkpoint header\n";
			return false;
		}

		// --- leaf parameter records ---
		leaf_params.clear();
		leaf_params.reserve(leafCount);
		for (uint32_t i = 0; i < leafCount; i++)
		{
			typename AutoGrad<DataType>::LeafParameterRecord rec;
			in.read(reinterpret_cast<char *>(&rec.param.start), sizeof(rec.param.start));
			in.read(reinterpret_cast<char *>(&rec.param.rows), sizeof(rec.param.rows));
			in.read(reinterpret_cast<char *>(&rec.param.cols), sizeof(rec.param.cols));

			uint32_t nameLen = 0;
			in.read(reinterpret_cast<char *>(&nameLen), sizeof(nameLen));
			if (nameLen > 0)
			{
				std::vector<char> nameBuf(nameLen);
				in.read(nameBuf.data(), static_cast<std::streamsize>(nameLen));
				rec.name = std::string(nameBuf.data(), nameLen);
			}

			if (!in)
			{
				err << "I/O error while reading leaf parameter records\n";
				return false;
			}

			leaf_params.push_back(std::move(rec));
		}

		// --- payload ---
		const auto byteCount = static_cast<std::streamsize>(paramCount * sizeof(DataType));

		values.resize(paramCount);
		grads.resize(paramCount);
		moment1.resize(paramCount);
		moment2.resize(paramCount);
		step_count = loadedStep;

		in.read(reinterpret_cast<char *>(values.data()), byteCount);
		in.read(reinterpret_cast<char *>(grads.data()), byteCount);
		in.read(reinterpret_cast<char *>(moment1.data()), byteCount);
		in.read(reinterpret_cast<char *>(moment2.data()), byteCount);

		if (!in)
		{
			err << "I/O error while reading checkpoint data\n";
			return false;
		}
		
		return true;
	}

	/// Convenience: save to a file path.
	bool saveToFile(const std::string &path) const
	{
		std::ofstream out(path, std::ios::binary);
		if (!out)
			return false;
		return save(out, std::cerr);
	}

	/// Convenience: load from a file path.
	bool loadFromFile(const std::string &path)
	{
		std::ifstream in(path, std::ios::binary);
		if (!in)
			return false;
		return load(in, std::cerr);
	}

	/// @}
};
