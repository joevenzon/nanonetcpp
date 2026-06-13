#pragma once

#include <cassert>

class NodeHandle
{
public:
	NodeHandle(int new_node_index = -1) : node(new_node_index) {}

	bool valid() const { return node >= 0; }
	int get_node_index() const { return node; }

private:
	int node;
};

struct ParameterHandle
{
	ParameterHandle() {}
	ParameterHandle(NodeHandle start_at, int num_rows, int num_cols) : start(start_at), rows(num_rows), cols(num_cols) {}

	NodeHandle start = NodeHandle(0);
	int rows = 0;
	int cols = 0;

	int size() const { return rows * cols; }
};