/*
 * Copyright (c) 2023, 2024, Oracle and/or its affiliates. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tribuo.util.buffers;

import java.nio.Buffer;
import java.util.Arrays;

/**
 * Base class for tensors wrapped around {@link Buffer} instances.
 * <p>
 * Contains a buffer and a shape. Most of the mutation methods live on the subclasses
 * to allow them to use primitive types. This class is mutable and exposes the buffer for
 * mutation. Remember to rewind it whenever you operate on the buffer using the implicit position methods.
 * <p>
 * Note it does not implement {@code org.tribuo.la.Tensor}, though it may do in some future version.
 * @param <B> The buffer type.
 */
public sealed abstract class TensorBuffer<B extends Buffer> permits FloatTensorBuffer, IntTensorBuffer, LongTensorBuffer {

    /**
     * The buffer holding the values.
     */
    protected final B buffer;
    /**
     * The shape of the tensor.
     */
    protected final long[] shape;
    /**
     * Stride values for indexing into the tensor.
     */
    protected final long[] strides;

    /**
     * The total number of elements in this tensor.
     */
    protected final int numElements;

    /**
     * Creates a TensorBuffer from the supplied buffer and shape.
     * @param buffer The buffer containing the data.
     * @param shape The shape.
     */
    public TensorBuffer(B buffer, long[] shape) {
        this.buffer = buffer;
        this.shape = shape;
        this.strides = new long[shape.length];
        this.strides[strides.length-1] = 1;
        for (int i = strides.length-1; i > 0; i--) {
            this.strides[i-1] = strides[i] * shape[i];
        }
        this.numElements = computeNumElements(shape);
        if (this.buffer.capacity() != this.numElements) {
            throw new IllegalArgumentException("Buffer has different capacity than the shape expects. Buffer.capacity = " + this.buffer.capacity() + ", numElements = " + this.numElements);
        }
    }

    /**
     * Access the buffer directly.
     * @return The buffer.
     */
    public B buffer() {
        return buffer;
    }

    /**
     * The shape.
     * @return The shape.
     */
    public long[] shape() {
        return shape;
    }

    /**
     * Deep copy of this tensor.
     * @return A copy of the tensor.
     */
    public abstract TensorBuffer<B> copy();

    /**
     * Computes the linear index from the supplied index array.
     * @param idxArr The index array.
     * @return The linear index into the buffer.
     */
    protected int computeIdx(long[] idxArr) {
        long idx = 0;
        for (int i = 0; i < shape.length; i++) {
            if (idxArr[i] >= shape[i]) {
                throw new IllegalArgumentException("Invalid index " + Arrays.toString(idxArr) + ", shape " + Arrays.toString(idxArr));
            }
            idx += idxArr[i] * strides[i];
        }
        return (int) idx;
    }

    /**
     * Computes the number of elements.
     * <p>
     * If we overflow the int it returns -1, and the tensor is invalid.
     * @param shape The shape.
     * @return The number of elements.
     */
    protected static int computeNumElements(long[] shape) {
        long total = 1;
        for (int i = 0; i < shape.length; i++) {
            long cur = shape[i];
            if ((((int) cur) != cur) || (cur < 0)) {
                total = -1;
                break;
            } else {
                total *= cur;
                if (total <= 0) {
                    break;
                }
            }
        }
        return (int) total;
    }
}
