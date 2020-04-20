//
// Created by jarek on 17.10.2019.
//

#include "boolean_array_2_d.hpp"
#include <math.h>
#include <stdexcept>

BooleanArray2D::BooleanArray2D(unsigned int rows, unsigned int cols)
    : rows(rows), cols(cols)
{
    if(rows == 0) {
        throw std::logic_error("Rows must be bigger than 0.");
    }
    if(cols == 0) {
        throw std::logic_error("Columns must be bigger than 0.");
    }

    unsigned int size = 8 * (unsigned int) ceil((float) (rows * cols) / 8.0);
    data = new unsigned char[size];
}

BooleanArray2D::~BooleanArray2D()
{
    delete[] data;
}

bool BooleanArray2D::at(unsigned int row, unsigned int col) const
{
    if(row >= rows) {
        throw std::out_of_range("Row out of range.");
    }
    if(col >= cols) {
        throw std::out_of_range("Column out of range.");
    }

    unsigned int index = row * cols + col;
    unsigned int byte_index = index / 8;
    unsigned int bit_index = index % 8;
    unsigned char byte = data[byte_index];
    //is bit with bit_index is set in byte:
    return (1 == ((byte >> bit_index) & 1U));
}

void BooleanArray2D::set(unsigned int row, unsigned int col, bool value)
{
    if(row >= rows) {
        throw std::out_of_range("Row out of range.");
    }
    if(col >= cols) {
        throw std::out_of_range("Column out of range.");
    }

    unsigned int index = row * cols + col;
    unsigned int byte_index = index / 8;
    unsigned int bit_index = index % 8;
    unsigned char &byte = data[byte_index];
    //set bit with bit_index in byte to 1 if value is true and to 0 if value is false:
    if(value) {
        byte |= 1U << bit_index;
    } else {
        byte &= ~(1U << bit_index);
    }
}
