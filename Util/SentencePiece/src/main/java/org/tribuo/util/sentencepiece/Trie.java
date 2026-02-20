/*
 * Copyright (c) 2023, 2026, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.util.sentencepiece;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

/**
 * Implements a double array character trie with the outputs packed into the ints storing the characters.
 */
public final class Trie {

    /**
     * Default maximum number of results returned by {@link #commonPrefixSearch(ByteBuffer)}.
     */
    public static final int DEFAULT_MAX_RESULTS = 32;

    private static final TrieResult NOT_FOUND = new TrieResult(-1, 0);

    private final IntBuffer buffer;

    Trie(Set<String> strings) {
        this.buffer = build(strings);
    }

    Trie(ByteBuffer buffer) {
        this.buffer = buffer.order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();
    }

    /**
     * Returns a copy of the byte buffer that backs this Trie.
     * @return A copy of the Trie buffer.
     */
    ByteBuffer getBuffer() {
        ByteBuffer output = ByteBuffer.allocate(buffer.limit() * Integer.BYTES);
        output.order(ByteOrder.LITTLE_ENDIAN);
        output.asIntBuffer().put(buffer);
        buffer.rewind();
        output.rewind();
        return output;
    }

    /**
     * Checks if the given byte sequence exists or not, if it exists its value and length
     * are returned. Otherwise, the returned record contains -1 and 0 respectively.
     * @param key The query.
     * @return The found value & length.
     */
    public TrieResult exactMatchSearch(byte[] key) {
        return exactMatchSearch(key, 0);
    }

    /**
     * Checks if the given byte sequence exists or not, if it exists its value and length
     * are returned. Otherwise, the returned record contains -1 and 0 respectively. The length
     * is from the supplied start position, not the start of the entity.
     * @param key The query.
     * @param startPos The start position in the trie.
     * @return The found value & length.
     */
    public TrieResult exactMatchSearch(byte[] key, int startPos) {
        int pos = startPos;
        int unit = buffer.get(pos);
        for (byte b : key) {
            pos ^= offset(unit) ^ b;
            unit = buffer.get(pos);
            if (label(unit) != b) {
                return NOT_FOUND;
            }
        }

        if (!hasLeaf(unit)) {
            return NOT_FOUND;
        }
        unit = buffer.get(pos ^ offset(unit));
        return new TrieResult(value(unit), key.length);
    }

    /**
     * Searches for all keys in the trie that are a prefix of the given byte buffer, starting
     * from position 0 in the trie and returning at most {@link #DEFAULT_MAX_RESULTS} results.
     * @param key The byte buffer to search from its current position.
     * @return The list of matching trie entries, each carrying its value and the length of the matching prefix.
     */
    public List<TrieResult> commonPrefixSearch(ByteBuffer key) {
        return commonPrefixSearch(key, 0);
    }

    /**
     * Searches for all keys in the trie that are a prefix of the given byte buffer, returning
     * at most {@link #DEFAULT_MAX_RESULTS} results.
     * @param key The byte buffer to search from its current position.
     * @param startPos The starting node position in the trie (0 for the root).
     * @return The list of matching trie entries, each carrying its value and the length of the matching prefix.
     */
    public List<TrieResult> commonPrefixSearch(ByteBuffer key, int startPos) {
        return commonPrefixSearch(key, startPos, DEFAULT_MAX_RESULTS);
    }

    /**
     * Searches the trie for keys which match a prefix of the given string.
     * @param query The character sequence to search.
     * @param startPos The start position in the key buffer.
     * @param maxNumResults The maximum number of results to return.
     * @return The keys which match the prefix of the query.
     */
    public List<TrieResult> commonPrefixSearch(ByteBuffer query, int startPos, int maxNumResults) {
        List<TrieResult> output = new ArrayList<>();
        int pos = startPos;

        int unit = buffer.get(pos);
        pos ^= offset(unit);
        for (int i = query.position(); i < query.remaining(); ++i) {
            pos ^= query.get(i);
            unit = buffer.get(pos);
            if (label(unit) != query.get(i)) {
                return output;
            }

            pos ^= offset(unit);
            if (hasLeaf(unit) && output.size() < maxNumResults) {
                output.add(new TrieResult(value(buffer.get(pos)), i+1));
            }
        }

        return output;
    }

    /**
     * Traverses the trie as a DFA. It starts in nodePos and transitions to
     * query[queryPos] in order, if there is no transition for query[queryPos] (i.e., the string
     * isn't in the trie) then it returns -2. Otherwise, it returns -1 if it has not
     * terminated traversal, or the non-negative value in the leaf of this query.
     * @param query The query to probe the DFA with.
     * @param nodePos The starting position in the DFA.
     * @param queryPos The starting position in the query.
     * @return A traversal result.
     */
    public TraverseResult traverse(byte[] query, int nodePos, int queryPos) {
        int id = nodePos;
        int unit = buffer.get(id);

        for (; queryPos < query.length; queryPos++) {
            id ^= offset(unit) ^ query[queryPos];
            unit = buffer.get(id);
            if (label(unit) != query[queryPos]) {
                return new TraverseResult(-2, id, queryPos);
            }
        }

        if (!hasLeaf(unit)) {
            return new TraverseResult(-1, id, queryPos);
        }
        unit = buffer.get(id ^ offset(unit));
        return new TraverseResult(value(unit), id, queryPos);
    }

    /**
     * Searches the trie for keys which match a prefix of the given buffer and return the longest one.
     * @param buffer The buffer to query.
     * @return The longest key which match the prefix of the query.
     */
    public PrefixMatch longestPrefixMatch(ByteBuffer buffer) {
        List<TrieResult> matches = commonPrefixSearch(buffer);

        if (matches.isEmpty()) {
            return new PrefixMatch(UTF8Utils.codepointLength(buffer.get(buffer.position())), false);
        } else {
            int length = 0;
            for (var res : matches) {
                length = Math.max(length, res.length());
            }
            return new PrefixMatch(length, true);
        }
    }

    /**
     * Checks if this input is immediately derived from the unit value or not.
     * @param input The unit value.
     * @return True if it is derived from the unit.
     */
    static boolean hasLeaf(int input) {
        return ((input >> 8) & 1) == 1;
    }

    /**
     * Returns the value stored in this leaf.
     * @param input The input leaf.
     * @return The value.
     */
    static int value(int input) {
        return (input & ((1 << 31) - 1));
    }

    /**
     * Returns the label associated with the input.
     * <p>
     * Leaves always return invalid labels which have the MSB set.
     * @param input The input.
     * @return The label
     */
    static int label(int input) {
        return input & ((1 << 31) | 0xFF);
    }

    /**
     * Returns the offset from the input to the derived values.
     * @param input The input.
     * @return The offset to the derived values.
     */
    static int offset(int input) {
        return (input >> 10) << ((input & (1 << 9)) >> 6);
    }

    private static IntBuffer build(Set<String> strings) {
        List<byte[]> keys = new ArrayList<>(strings.size());
        for (var s : strings) {
            keys.add(s.getBytes(UTF8Utils.UTF8));
        }
        // Keys must be sorted by unsigned byte order for the double-array trie
        keys.sort(Trie::compareUnsignedBytes);

        TrieBuilder builder = new TrieBuilder();
        builder.build(keys);
        return builder.toIntBuffer();
    }

    private static int compareUnsignedBytes(byte[] a, byte[] b) {
        int minLen = Math.min(a.length, b.length);
        for (int i = 0; i < minLen; i++) {
            int diff = (a[i] & 0xFF) - (b[i] & 0xFF);
            if (diff != 0) {
                return diff;
            }
        }
        return a.length - b.length;
    }

    /**
     * Double-array trie builder, translated from Darts-clone (darts.h).
     * <p>
     * This implements the build_from_keyset path used when no explicit values
     * are provided (each key's index in the sorted array serves as its value).
     */
    private static final class TrieBuilder {
        private static final int BLOCK_SIZE = 256;
        private static final int NUM_EXTRA_BLOCKS = 16;
        private static final int NUM_EXTRAS = BLOCK_SIZE * NUM_EXTRA_BLOCKS;
        private static final int UPPER_MASK = 0xFF << 21;
        private static final int LOWER_MASK = 0xFF;

        // Circular buffer of size NUM_EXTRAS
        private final int[] extraPrev;
        private final int[] extraNext;
        private final boolean[] extraIsFixed;
        private final boolean[] extraIsUsed;

        // Temporary labels collected during arrange
        private final List<Integer> labels = new ArrayList<>();

        private int[] units;
        private int unitsSize;
        private int extrasHead;

        private boolean built = false;

        TrieBuilder() {
            units = new int[BLOCK_SIZE];
            unitsSize = 0;
            extrasHead = 0;
            extraPrev = new int[NUM_EXTRAS];
            extraNext = new int[NUM_EXTRAS];
            extraIsFixed = new boolean[NUM_EXTRAS];
            extraIsUsed = new boolean[NUM_EXTRAS];
        }

        private void setLeaf(int id) {
            units[id] |= 1 << 8;
        }

        private void setValue(int id, int value) {
            units[id] = value | (1 << 31);
        }

        private void setUnitLabel(int id, int label) {
            units[id] = (units[id] & ~0xFF) | (label & 0xFF);
        }

        private void setOffset(int id, int offset) {
            if (Integer.toUnsignedLong(offset) >= (1L << 29)) {
                throw new IllegalStateException("failed to modify unit: too large offset");
            }
            units[id] &= (1 << 31) | (1 << 8) | 0xFF;
            if (offset < (1 << 21)) {
                units[id] |= (offset << 10);
            } else {
                units[id] |= (offset << 2) | (1 << 9);
            }
        }

        private int extraIdx(int id) {
            return id % NUM_EXTRAS;
        }

        private int numBlocks() {
            return unitsSize / BLOCK_SIZE;
        }

        private void ensureCapacity(int capacity) {
            if (capacity > units.length) {
                units = Arrays.copyOf(units, capacity);
            }
        }

        void build(List<byte[]> keys) {
            int numUnits = 1;
            while (numUnits < keys.size()) {
                numUnits = numUnits << 1;
            }
            ensureCapacity(numUnits);

            reserveId(0);
            extraIsUsed[extraIdx(0)] = true;
            setOffset(0, 1);
            setUnitLabel(0, 0);

            if (!keys.isEmpty()) {
                buildFromKeyset(keys, 0, keys.size(), 0, 0);
            }

            fixAllBlocks();
            built = true;
        }

        IntBuffer toIntBuffer() {
            if (!built) {
                throw new IllegalStateException("Trie must be built before the buffer can be extracted.");
            } else {
                IntBuffer buf = IntBuffer.allocate(unitsSize);
                buf.put(units, 0, unitsSize);
                buf.flip();
                return buf;
            }
        }

        private void buildFromKeyset(List<byte[]> keys, int begin, int end, int depth, int dicId) {
            int offset = arrangeFromKeyset(keys, begin, end, depth, dicId);

            // Skip past keys that end at this depth
            for (; begin < end; begin++) {
                if (keyByte(keys, begin, depth) != 0) {
                    break;
                }
            }
            if (begin == end) {
                return;
            }

            int lastBegin = begin;
            int lastLabel = keyByte(keys, begin, depth);
            begin++;
            for (int i = begin; i < end; i++) {
                int label = keyByte(keys, i, depth);
                if (label != lastLabel) {
                    buildFromKeyset(keys, lastBegin, i, depth + 1, offset ^ lastLabel);
                    lastBegin = i;
                    lastLabel = keyByte(keys, i, depth);
                }
            }
            buildFromKeyset(keys, lastBegin, end, depth + 1, offset ^ lastLabel);
        }

        private int arrangeFromKeyset(List<byte[]> keys, int begin, int end, int depth, int dicId) {
            labels.clear();

            int value = -1;
            for (int i = begin; i < end; i++) {
                int label = keyByte(keys, i, depth);
                if (label == 0) {
                    if (value == -1) {
                        value = i;
                    }
                }

                if (labels.isEmpty()) {
                    labels.add(label);
                } else if (label != labels.getLast()) {
                    if (label < labels.getLast()) {
                        throw new IllegalStateException("failed to build trie: wrong key order");
                    }
                    labels.add(label);
                }
            }

            int offset = findValidOffset(dicId);
            setOffset(dicId, dicId ^ offset);

            for (Integer label : labels) {
                int dicChildId = offset ^ label;
                reserveId(dicChildId);
                if (label == 0) {
                    setLeaf(dicId);
                    setValue(dicChildId, value);
                } else {
                    setUnitLabel(dicChildId, label);
                }
            }
            extraIsUsed[extraIdx(offset)] = true;

            return offset;
        }

        private int findValidOffset(int id) {
            if (extrasHead >= unitsSize) {
                return unitsSize | (id & LOWER_MASK);
            }

            int unfixedId = extrasHead;
            do {
                int offset = unfixedId ^ labels.getFirst();
                if (isValidOffset(id, offset)) {
                    return offset;
                }
                unfixedId = extraNext[extraIdx(unfixedId)];
            } while (unfixedId != extrasHead);

            return unitsSize | (id & LOWER_MASK);
        }

        private boolean isValidOffset(int id, int offset) {
            if (extraIsUsed[extraIdx(offset)]) {
                return false;
            }

            int relOffset = id ^ offset;
            if ((relOffset & LOWER_MASK) != 0 && (relOffset & UPPER_MASK) != 0) {
                return false;
            }

            for (int i = 1; i < labels.size(); i++) {
                if (extraIsFixed[extraIdx(offset ^ labels.get(i))]) {
                    return false;
                }
            }

            return true;
        }

        private void reserveId(int id) {
            if (id >= unitsSize) {
                expandUnits();
            }

            int idx = extraIdx(id);
            if (id == extrasHead) {
                extrasHead = extraNext[idx];
                if (extrasHead == id) {
                    extrasHead = unitsSize;
                }
            }
            int prevIdx = extraIdx(extraPrev[idx]);
            int nextIdx = extraIdx(extraNext[idx]);
            extraNext[prevIdx] = extraNext[idx];
            extraPrev[nextIdx] = extraPrev[idx];
            extraIsFixed[idx] = true;
        }

        private void expandUnits() {
            int srcNumUnits = unitsSize;
            int srcNumBlocks = numBlocks();

            int destNumUnits = srcNumUnits + BLOCK_SIZE;
            int destNumBlocks = srcNumBlocks + 1;

            if (destNumBlocks > NUM_EXTRA_BLOCKS) {
                fixBlock(srcNumBlocks - NUM_EXTRA_BLOCKS);
            }

            // Grow backing array if needed
            if (destNumUnits > units.length) {
                units = Arrays.copyOf(units, Math.max(units.length * 2, destNumUnits));
            }
            unitsSize = destNumUnits;

            if (destNumBlocks > NUM_EXTRA_BLOCKS) {
                for (int id = srcNumUnits; id < destNumUnits; id++) {
                    int idx = extraIdx(id);
                    extraIsUsed[idx] = false;
                    extraIsFixed[idx] = false;
                }
            }

            // Link new block's units into a chain
            for (int i = srcNumUnits + 1; i < destNumUnits; i++) {
                extraNext[extraIdx(i - 1)] = i;
                extraPrev[extraIdx(i)] = i - 1;
            }
            extraPrev[extraIdx(srcNumUnits)] = destNumUnits - 1;
            extraNext[extraIdx(destNumUnits - 1)] = srcNumUnits;

            // Splice new chain into existing free list before extrasHead
            int oldHeadPrev = extraPrev[extraIdx(extrasHead)];
            extraPrev[extraIdx(srcNumUnits)] = oldHeadPrev;
            extraNext[extraIdx(destNumUnits - 1)] = extrasHead;
            extraNext[extraIdx(oldHeadPrev)] = srcNumUnits;
            extraPrev[extraIdx(extrasHead)] = destNumUnits - 1;
        }

        private void fixAllBlocks() {
            int begin = 0;
            if (numBlocks() > NUM_EXTRA_BLOCKS) {
                begin = numBlocks() - NUM_EXTRA_BLOCKS;
            }
            int end = numBlocks();

            for (int blockId = begin; blockId != end; blockId++) {
                fixBlock(blockId);
            }
        }

        private void fixBlock(int blockId) {
            int begin = blockId * BLOCK_SIZE;
            int end = begin + BLOCK_SIZE;

            // Find first unused offset in this block
            int unusedOffset = 0;
            for (int offset = begin; offset != end; offset++) {
                if (!extraIsUsed[extraIdx(offset)]) {
                    unusedOffset = offset;
                    break;
                }
            }

            // Fix all unfixed units in this block
            for (int id = begin; id != end; id++) {
                if (!extraIsFixed[extraIdx(id)]) {
                    reserveId(id);
                    setUnitLabel(id, (id ^ unusedOffset) & 0xFF);
                }
            }
        }

        private static int keyByte(List<byte[]> keys, int keyId, int depth) {
            byte[] key = keys.get(keyId);
            if (depth >= key.length) {
                return 0;
            } else {
                return key[depth] & 0xFF;
            }
        }
    }

    /**
     * A tuple representing a prefix match, with the number of characters consumed and if it actually matched.
     * @param lengthConsumed The number of characters consumed by the match.
     * @param found If it actually matched.
     */
    public record PrefixMatch(int lengthConsumed, boolean found) {}

    /**
     * The result of probing the trie to completion.
     * @param value The value of the terminal node.
     * @param length The number of characters consumed.
     */
    public record TrieResult(int value, int length) {}

    /**
     * The result of calling {@link #traverse(byte[], int, int)}.
     * @param value The value of the terminal node, -1 if it didn't complete traversal, or -2 if the query isn't in the trie.
     * @param nodePos The final position in the trie.
     * @param queryPos The final position in the query array.
     */
    public record TraverseResult(int value, int nodePos, int queryPos) {}

}
