#pragma once

#include "plainarray.h"

#define MAX_TENSOR_DIM 8

struct TensorShape
{
	TensorShape() = default;
	TensorShape(std::initializer_list<int> d) : dims(d) {}

	bool operator==(const TensorShape & other) const
	{
		// 1. If they have different numbers of dimensions, they cannot be equal
		if (rank() != other.rank())
			return false;

		// 2. Compare the elements within the active range of the PlainArray
		// std::equal compares the range [begin, end) of the first 
		// against the beginning of the second.
		return std::equal(dims.begin(), dims.end(), other.dims.begin());
	}

	bool operator!=(const TensorShape & other) const
	{
		return !(*this == other);
	}

	size_t numel() const
	{
		size_t result = 1;
		for (int dim : dims)
		{
			result *= dim;
		}
		return result;
	}

	int rank() const { return (int)dims.size(); }

	int dim(int dimension) const { return dimension < dims.size() ? dims[dimension] : 1; }

	PlainArray <int, MAX_TENSOR_DIM> dims;
};

template <typename DataType>
class Tensor
{
public:
	Tensor() = default;
	Tensor(std::span <DataType> new_value_buffer, std::span <DataType> new_grad_buffer, TensorShape s) : value_buffer(new_value_buffer), grad_buffer(new_grad_buffer), shape(s)
	{
		assert(value_buffer.size() == shape.numel() && "Value buffer size much match shape");
		assert(grad_buffer.size() == value_buffer.size() && "Grad buffer size must match value buffer");
	}

	size_t numel() const { return value_buffer.size(); }
	const TensorShape & get_shape() const { return shape; }

	void zero_grad()
	{
		std::fill(grad_buffer.begin(), grad_buffer.end(), static_cast<DataType>(0));
	}

	std::span<DataType> values() { return value_buffer; }
	std::span<const DataType> values() const { return value_buffer; }

	std::span<DataType> gradients() { return grad_buffer; }
	std::span<const DataType> gradients() const { return grad_buffer; }

	// Non-const accessor for reading and writing: tensor(i, j, k) = value;
	template <typename... Indices>
	DataType & operator()(Indices... indices)
	{
		return value_buffer[get_flat_index(indices...)];
	}

	// Const accessor for read-only: DataType v = tensor(i, j, k);
	template <typename... Indices>
	const DataType & operator()(Indices... indices) const
	{
		return value_buffer[get_flat_index(indices...)];
	}

private:
	// Helper to convert multi-dimensional coordinates into a flat array index
	// Formula: index = (i * stride_0) + (j * stride_1) + (k * stride_2) ...
	template <typename... Indices>
	size_t get_flat_index(Indices... indices) const
	{
		// Put the variadic arguments into a temporary array for easy iteration
		int coords[] = { static_cast<int>(indices)... };

		// Basic safety check: ensure user provided the correct number of indices for the rank
		assert((sizeof...(indices) == (size_t)shape.rank()) && "Number of indices must match tensor rank");

		size_t flat_idx = 0;
		size_t stride = 1;

		// To calculate the flat index in row-major order (PyTorch style), 
		// we iterate from the last dimension backwards.
		for (int i = shape.rank() - 1; i >= 0; --i)
		{
			// Boundary check
			assert(coords[i] >= 0 && coords[i] < shape.dims[i] && "Index out of bounds");

			flat_idx += coords[i] * stride;
			stride *= shape.dims[i];
		}

		return flat_idx;
	}

	std::span <DataType> value_buffer;
	std::span <DataType> grad_buffer;
	TensorShape shape;
};
