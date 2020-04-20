//
// Created by jarek on 17.10.2019.
//

#ifndef BOOLEAN_ARRAY_2_D_HPP
#define BOOLEAN_ARRAY_2_D_HPP

#include <bitset>

/**
 * @brief Class representing 2D array of booleans. Booleans are represented as single bits for memory efficiency.
 */
class BooleanArray2D {
protected:
    unsigned char *data;
public:
    unsigned int rows;
    unsigned int cols;
    BooleanArray2D(unsigned int rows, unsigned int cols);
    ~BooleanArray2D();
    [[nodiscard]] bool at(unsigned int row, unsigned int col) const;
    void set(unsigned int row, unsigned int col, bool value);
};


#endif //BOOLEAN_ARRAY_2_D_HPP
