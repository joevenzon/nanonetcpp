#pragma once

#include <cassert>

class TensorHandle
{
public:
	TensorHandle(int new_node_index = -1) : node(new_node_index) {}

	bool valid() const { return node >= 0; }
	int get_node_index() const { return node; }

private:
	int node;
};
