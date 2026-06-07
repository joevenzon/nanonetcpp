#pragma once

#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "autograd.h"

#define CHECKPOINT_MAGIC "NNCHK001"
#define CHECKPOINT_VERSION 1

template <typename DataType>
struct ParameterCheckpoint
{
	// the model weights themselves
	std::vector <DataType> values;

	// for gradient accumulation
	std::vector<DataType> grads;

	// for the optimizer
	std::vector<DataType> moment1;		// Adam first moment
	std::vector<DataType> moment2;		// Adam second moment
	int step_count = 0;					// for Adam bias correction

	void init(AutoGrad<DataType> & grad)
	{
		values.resize(grad.size(), 0);
		grads.resize(grad.size(), 0);
		moment1.resize(grad.size(), 0);
		moment2.resize(grad.size(), 0);

		for (int i = 0; i < grad.size(); i++)
		{
			values[i] = grad.get(NodeHandle(i)).data;
		}

		step_count = 0;
	}

	void update(AutoGrad<DataType> & grad)
	{
		for (int i = 0; i < values.size(); i++)
		{
			const AutoGrad<DataType>::Node & node = grad.get(NodeHandle(i));
			values[i] = node.data;
			grads[i] = node.grad;
		}
	}

	size_t size() const { return values.size(); }

	/// Save checkpoint to an ostream in binary format.
	///
	/// Format:
	///   Header (32 bytes)
	///     - Magic:      8 bytes  "NNCHK001" or similar
	///     - Version:    4 bytes  uint32_t (little-endian)
	///     - DataTypeID: 4 bytes  uint32_t (1 = 32-bit float)
	///     - ParamCount: 8 bytes  uint64_t (number of parameters)
	///     - StepCount:  8 bytes  int (signed, for Adam bias correction)
	///     - Reserved:   4 bytes  zero padding for future use
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
			err << "Cannot save an empty checkpoint\n";
			return false;
		}
		out.write(reinterpret_cast<const char *>(&dataTypeId), sizeof(dataTypeId));

		uint64_t paramCount = static_cast<uint64_t>(values.size());
		out.write(reinterpret_cast<const char *>(&paramCount), sizeof(paramCount));

		out.write(reinterpret_cast<const char *>(&step_count), sizeof(step_count));

		// reserved padding
		uint32_t reserved = 0;
		out.write(reinterpret_cast<const char *>(&reserved), sizeof(reserved));

		if (!out)
		{
			err << "I/O error while writing checkpoint header";
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
			err << "Unsupported checkpoint version: " + version << "\n";
			return false;
		}

		uint32_t dataTypeId = 0;
		in.read(reinterpret_cast<char *>(&dataTypeId), sizeof(dataTypeId));
		bool typeMatch = false;
		if constexpr (std::is_same_v<DataType, float>)
			typeMatch = (dataTypeId == 1);
		if (!typeMatch)
		{
			err << "DataType mismatch: file stores id=" + dataTypeId << " which is unsupported\n";
			return false;
		}

		uint64_t paramCount = 0;
		in.read(reinterpret_cast<char *>(&paramCount), sizeof(paramCount));

		int loadedStep = 0;
		in.read(reinterpret_cast<char *>(&loadedStep), sizeof(loadedStep));

		// skip reserved padding
		char reserved[4];
		in.read(reserved, sizeof(reserved));

		if (!in)
		{
			err << "I/O error while reading checkpoint header\n";
			return false;
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
