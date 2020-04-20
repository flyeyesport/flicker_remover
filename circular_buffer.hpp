//
// Created by jarek on 14.05.17.
//

#ifndef CIRCULAR_BUFFER_HPP
#define CIRCULAR_BUFFER_HPP


template<class T>
class CircularBuffer;


/**
 * @brief Simple, constant length buffer of pointers to elements of type T.
 *
 * Pointers of elements of type <b>T</b> can be removed from the beginning of buffer with <b>pop()</b> method and added
 * at the end of the buffer with <b>push()</b> method. Pushing a pointer to the item to the end of a full buffer
 * automatically pops and returns pointer to first element in buffer. This way number of elements in buffer never
 * exceeds maximum buffer length.
 *
 * @tparam T A type of elements to which pointers will be stored in buffer. It can be any type.
 */
template<class T>
class CircularBuffer<T *> {
protected:

    /**
     * Actual number of pointers to elements stored in the buffer.
     */
    unsigned int count_of_elements;

    /**
     * The length of the buffer. Maximum number of pointers to elements that can be stored in the buffer.
     */
    unsigned int count_of_slots;

    /**
     * Index of the first non-NULL element int the buffer. <b>First</b> means element in the buffer that was added
     * before all other elements in the buffer. Since buffer is circular its value can be bigger, equal to, or smaller
     * than <b>last_element_index</b>.
     */
    unsigned int first_element_index;

    /**
     * Index of the last non-NULL element in the buffer. <b>Last</b> means element in the buffer that was added after
     * all other elements in the buffer.
     */
    unsigned int last_element_index;

    /**
     *  Internal data structure. Simple table with as many elements of <b>T *</b> type as the length of the buffer.
     */
    T **data;
public:
    /**
     * @brief Buffer constructor.
     *
     * Initializes internal structures and allocates memory to hold <b>size</b> number of pointers to elements of
     * type T.
     * @param size Length of the buffer. At most that many pointers to elements of type T will be stored in the buffer.
     */
    explicit CircularBuffer(unsigned int size);

    /**
     * @brief Buffer destructor.
     *
     * Frees memory of the internal buffer, but does not free memory pointed by pointers to elements.
     */
    virtual ~CircularBuffer();

    /**
     * @brief Inserts pointer to element at the end of the buffer.
     *
     * If the buffer is full it removes and returns pointer to first element (to element added before all other
     * elements).
     * @param element Pointer to element that will be stored in the buffer.
     * @return Pointer to removed first element from the buffer when it was full or <b>nullptr</b> when buffer was not
     * full before adding pointer to the element.
     */
    virtual T *push(T *element);

    /**
     * @brief Removes and returns pointer to the first element in the buffer.
     *
     * Number of stored elements in the buffer is decreased.
     * @return Removed pointer from the buffer or <b>nullptr</b> when the buffer is empty.
     */
    virtual T *pop();

    /**
     * @brief Random access operator.
     *
     * Returns pointer to element with <b>index</b>. <b>First_element_index</b> is treated as index 0 and position of
     * returned pointer to element is counted from first_element_index.
     * @param index Number of returned element starting form 0 as <b>first_element_index</b>. It should be less than
     * length of the buffer. It can be a negative value: -1 means last_element_index, -2 means one before
     * last_element_index and so on. The range is still checked, for example: if buffer has maximum 10 spots for
     * elements and 3 elements, then -1 will return last element, -3 will return first element, and -4 will return
     * nullptr.
     * @return Pointer to the element stored at <b>index</b> position counting from <b>first_element_index</b>. If index
     * is out of bounds (less than 0 or equal to or more than length of the buffer) or there is no pointer at
     * <b>index</b> in the buffer, <b>nullptr</b> is returned.
     */
    T *operator[](int index) const;

    /**
     * @brief Updater of one of the values in the buffer.
     *
     * Updates/changes value pointed by element at position <b>index</b> to new value passed in the parameter.
     * @param index Number of changed element starting form 0 as <b>first_element_index</b>. It should be less than
     * length of the buffer.
     * @param value Pointer of new value that will be assigned to the element stored at <b>index</b>.
     *
     * @return Pointer to stored in position at index counting from <b>first_element_index</b> before update. It should
     * be used to eventually release memory.
     */
    T *update(int index, T *value);

    /**
     * @brief Returns pointer to the last element in the buffer.
     *
     * Pointer to element added after all other elements in the buffer is returned or <b>nullptr</b> when the buffer
     * is empty.
     * @return Pointer to the last element in the buffer is returned or <b>nullptr</b> when the buffer is empty.
     */
    T *last() const;

    /**
     * @brief Returns pointer to the first element in the buffer.
     *
     * Pointer to element added before all other elements in the buffer is returned or <b>nullptr</b> when the buffer
     * is empty.
     * @return Pointer to the first element in the buffer is returned or <b>nullptr</b> when the buffer is empty.
     */
    T *first() const;

    /**
     * @brief Number of pointers to elements stored in the buffer.
     *
     * @return Actual number of pointers to elements stored in the buffer.
     */
    unsigned int size() const;

    /**
     * @brief Maximum number of pointers to elements that can be stored in the buffer.
     *
     * @return Maximum number of pointers to elements that can be stored in the buffer.
     */
    unsigned int maxSize() const;

    /**
     * @brief Returns information if buffer is fully filled with pointers to elements.
     *
     * @return True if number of stored pointers to elements is equal to the length of the buffer, false otherwise.
     */
    bool isFull() const;

    /**
     * @brief Returns information if buffer is empty.
     *
     * @return True if number of stored pointers to elements is equal to zero, false otherwise.
     */
    bool isEmpty() const;

    /**
     * @brief Sets all pointers of data to nullptrs. Memory pointed previously by pointers is not freed!
     */
    void clear();

    /**
     * @brief Frees internal buffer and allocates it again with the new size. All internal data is reset.
     * Memory pointed by pointers stored in the buffer is not freed.
     *
     * @param new_size New size of the internal data buffer.
     */
    void setMaxSize(unsigned int new_size);
};


//------------------------------ IMPLEMENTATION ------------------------------

template<class T>
CircularBuffer<T *>::CircularBuffer(unsigned int size)
    : count_of_slots(size)
{
    count_of_elements = 0;
    first_element_index = 0; //initial value is not important, these values are always set when elements are
    last_element_index = 0; //added or removed to/from buffer
    //we assume here that count_of_slots is bigger than 0
    if(count_of_slots > 0) {
        data = new T *[count_of_slots];
        for(unsigned int i = 0; i < count_of_slots; i++) {
            data[i] = nullptr;
        }
    } else {
        data = nullptr;
    }
}


template<class T>
CircularBuffer<T *>::~CircularBuffer()
{
    if(count_of_slots > 0) {
        delete[] data;
    }
}


template<class T>
T *CircularBuffer<T *>::push(T *element)
{
    T *ret;
    if(count_of_slots == count_of_elements) {
        //remove first element to make room for the new one, return it so the caller can deal with removed element
        ret = pop();
    } else {
        ret = nullptr;
    }
    if(count_of_elements == 0) {
        first_element_index = 0;
        last_element_index = 0;
    } else {
        if(last_element_index == count_of_slots - 1) {
            last_element_index = 0;
        } else {
            last_element_index++;
        }
    }
    data[last_element_index] = element;
    count_of_elements++;
    return ret;
}


template<class T>
T *CircularBuffer<T *>::pop()
{
    //remove first element from buffer and return it to the caller
    if(count_of_elements == 0) {
        return nullptr;
    }

    T *ret = data[first_element_index];
    data[first_element_index] = nullptr;
    if(first_element_index == count_of_slots - 1) {
        first_element_index = 0;
    } else {
        first_element_index++;
    }
    count_of_elements--;
    return ret;
}


template<class T>
T *CircularBuffer<T *>::operator[](int index) const
{
    if(count_of_elements == 0 || index > ((int) count_of_elements) - 1 || index < -1 * ((int) count_of_elements)) {
        return nullptr;
    }
    int real_index;
    if(index >= 0) {
        real_index = (first_element_index + index) % count_of_slots;
    } else {
        //we add count_of_slots here to be sure that left side of operator % is positive.
        real_index = (((int) last_element_index) + count_of_slots + 1 + index) % count_of_slots;
    }
    return data[real_index];
}


template<class T>
T *CircularBuffer<T *>::last() const
{
    if(count_of_elements == 0) {
        return nullptr;
    }
    return data[last_element_index];
}


template<class T>
T *CircularBuffer<T *>::first() const
{
    if(count_of_elements == 0) {
        return nullptr;
    }
    return data[first_element_index];
}


template<class T>
unsigned int CircularBuffer<T *>::size() const
{
    return count_of_elements;
}

template<class T>
unsigned int CircularBuffer<T *>::maxSize() const
{
    return count_of_slots;
}

template<class T>
bool CircularBuffer<T *>::isFull() const
{
    return (count_of_elements == count_of_slots);
}

template<class T>
bool CircularBuffer<T *>::isEmpty() const
{
    return (count_of_elements == 0);
}

template<class T>
T *CircularBuffer<T *>::update(int index, T *value)
{
    int real_index = (first_element_index + index) % count_of_slots;
    T *ret = data[real_index];
    data[real_index] = value;
    return ret;
}

template<class T>
void CircularBuffer<T *>::clear()
{
    count_of_elements = 0;
    first_element_index = 0; //initial value is not important, these values are always set when elements are
    last_element_index = 0; //added or removed to/from buffer
    for(int i = 0; i < count_of_slots; i++) {
        data[i] = nullptr;
    }
}

template<class T>
void CircularBuffer<T *>::setMaxSize(unsigned int new_size)
{
    delete[] data;
    count_of_elements = 0;
    first_element_index = 0; //initial value is not important, these values are always set when elements are
    last_element_index = 0; //added or removed to/from buffer
    //we assume here that count_of_slots is bigger than 0
    count_of_slots = new_size;
    if(count_of_slots > 0) {
        data = new T *[count_of_slots];
        for(unsigned int i = 0; i < count_of_slots; i++) {
            data[i] = nullptr;
        }
    } else {
        data = nullptr;
    }
}

#endif //CIRCULAR_BUFFER_HPP
