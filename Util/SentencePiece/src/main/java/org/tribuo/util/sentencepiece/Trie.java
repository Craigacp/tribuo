/*
 * Copyright (c) 2023, Oracle and/or its affiliates. All rights reserved.
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
import java.util.List;
import java.util.Set;

public final class Trie {

    private static final TrieResult NOT_FOUND = new TrieResult(-1, 0);

    private final IntBuffer buffer;

    Trie(Set<String> strings) {
        this.buffer = build(strings);
    }

    Trie(ByteBuffer buffer) {
        this.buffer = buffer.order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();
    }

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
        for (int i = 0; i < key.length; ++i) {
            pos ^= offset(unit) ^ key[i];
            unit = buffer.get(pos);
            if (label(unit) != key[i]) {
                return NOT_FOUND;
            }
        }

        if (!hasLeaf(unit)) {
            return NOT_FOUND;
        }
        unit = buffer.get(pos ^ offset(unit));
        return new TrieResult(value(unit), key.length);
    }

    public List<TrieResult> commonPrefixSearch(byte[] key) {
        return commonPrefixSearch(key, 0);
    }

    public List<TrieResult> commonPrefixSearch(byte[] key, int startPos) {
        return commonPrefixSearch(key, startPos, 32);
    }

    /**
     * Searches the trie for keys which match a prefix of the given string.
     * @param key
     * @param startPos
     * @param maxNumResults
     * @return
     */
    public List<TrieResult> commonPrefixSearch(byte[] key, int startPos, int maxNumResults) {
        List<TrieResult> output = new ArrayList<>();
        int pos = startPos;

        int unit = buffer.get(pos);
        pos ^= offset(unit);
        for (int i = 0; i < key.length; ++i) {
            pos ^= key[i];
            unit = buffer.get(pos);
            if (label(unit) != key[i]) {
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
     * key[keyPos] in order, if there is no transition for key[keyPos] (i.e. the string
     * isn't in the trie) then it returns -2. Otherwise, it returns -1 if it has not
     * terminated traversal, or the non-negative value in the leaf of this key.
     * @param key
     * @param nodePos
     * @param keyPos
     * @return
     */
    public TraverseResult traverse(byte[] key, int nodePos, int keyPos) {
        int id = nodePos;
        int unit = buffer.get(id);

        for ( ; keyPos < key.length; ++keyPos) {
            id ^= offset(unit) ^ key[keyPos];
            unit = buffer.get(id);
            if (label(unit) != key[keyPos]) {
                return new TraverseResult(-2, id, keyPos);
            }
        }

        if (!hasLeaf(unit)) {
            return new TraverseResult(-1, id, keyPos);
        }
        unit = buffer.get(id ^ offset(unit));
        return new TraverseResult(value(unit), id, keyPos);
    }

    public PrefixMatch longestPrefixMatch(ByteBuffer buffer) {
        List<Trie.TrieResult> matches = commonPrefixSearch(buffer);

        if (matches.isEmpty()) {
            return new PrefixMatch(SPModel.utf8CodepointLength(buffer.get(buffer.position())), false);
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
        List<ByteBuffer> bufs = new ArrayList<>(strings.size());
        for (var s : strings) {
            bufs.add(SPModel.UTF8.encode(s));
        }

    }

    public record PrefixMatch(int lengthConsumed, boolean found) {}

    public record TrieResult(int value, int length) {}

    public record TraverseResult(int value, int nodePos, int keyPos) {}

}
