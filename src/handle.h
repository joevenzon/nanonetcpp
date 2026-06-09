#pragma once

#include <cassert>

typedef int NodeHandle;

struct NodeMatrixHandle
{
	NodeMatrixHandle() {}
	NodeMatrixHandle(NodeHandle start_at, int num_rows, int num_cols) : start(start_at), rows(num_rows), cols(num_cols) {}

	NodeHandle start = 0;
	int rows = 0;
	int cols = 0;

	int size() const { return rows * cols; }
};