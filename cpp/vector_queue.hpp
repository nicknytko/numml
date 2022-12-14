#ifndef VECTOR_QUEUE_HPP_
#define VECTOR_QUEUE_HPP_

#include <cstdint>
#include <vector>

namespace numml {
    template <typename T>
    class vector_queue {
    private:
        std::vector<T> v;
        int64_t capacity;
        int64_t start;
        int64_t end;

    public:
        vector_queue(const int64_t size): v(size, T()), capacity(size), start(0), end(0) {};

        void push_back(const T& t) {
            assert((start % capacity + 1) != (end % capacity));

            v[end % capacity] = t;
            end++;
        }
        T pop_front() {
            assert(!empty());

            T temp = v[start % capacity];
            start++;
            return temp;
        }
        T peek_front() const {
            return v[start % capacity];
        }
        int64_t size() const {
            return end - start;
        }
        bool empty() const {
            return end == start;
        }
        void clear() {
            start = 0;
            end = 0;
        }
    };
}

#endif
