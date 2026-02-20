/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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

import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.TreeSet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TrieTest {

    @Test
    public void testSingleKey() {
        Trie trie = new Trie(Set.of("hello"));
        byte[] key = "hello".getBytes(StandardCharsets.UTF_8);

        Trie.TrieResult result = trie.exactMatchSearch(key);
        assertEquals(0, result.value());
        assertEquals(key.length, result.length());
    }

    @Test
    public void testExactMatchFound() {
        Set<String> words = Set.of("apple", "banana", "cherry", "date", "fig");
        Trie trie = new Trie(words);

        for (String word : words) {
            byte[] key = word.getBytes(StandardCharsets.UTF_8);
            Trie.TrieResult result = trie.exactMatchSearch(key);
            assertTrue(result.value() >= 0,
                    "Expected to find '" + word + "' but got value " + result.value());
            assertEquals(key.length, result.length());
        }
    }

    @Test
    public void testExactMatchNotFound() {
        Set<String> words = Set.of("apple", "banana", "cherry");
        Trie trie = new Trie(words);

        byte[] missing = "grape".getBytes(StandardCharsets.UTF_8);
        Trie.TrieResult result = trie.exactMatchSearch(missing);
        assertEquals(-1, result.value());
    }

    @Test
    public void testPrefixNotMatchedAsExact() {
        Set<String> words = Set.of("apple", "application");
        Trie trie = new Trie(words);

        // "app" is a prefix of both but is not in the trie
        byte[] prefix = "app".getBytes(StandardCharsets.UTF_8);
        Trie.TrieResult result = trie.exactMatchSearch(prefix);
        assertEquals(-1, result.value());
    }

    @Test
    public void testSharedPrefixes() {
        Set<String> words = Set.of("a", "ab", "abc", "abcd");
        Trie trie = new Trie(words);

        for (String word : words) {
            byte[] key = word.getBytes(StandardCharsets.UTF_8);
            Trie.TrieResult result = trie.exactMatchSearch(key);
            assertTrue(result.value() >= 0, "Expected to find '" + word + "'");
            assertEquals(key.length, result.length());
        }
    }

    @Test
    public void testDistinctValuesForDistinctKeys() {
        // Values are assigned as the index in the sorted key array
        Set<String> words = Set.of("alpha", "beta", "gamma");
        Trie trie = new Trie(words);

        byte[] alphaKey = "alpha".getBytes(StandardCharsets.UTF_8);
        byte[] betaKey = "beta".getBytes(StandardCharsets.UTF_8);
        byte[] gammaKey = "gamma".getBytes(StandardCharsets.UTF_8);

        int alphaVal = trie.exactMatchSearch(alphaKey).value();
        int betaVal = trie.exactMatchSearch(betaKey).value();
        int gammaVal = trie.exactMatchSearch(gammaKey).value();

        assertEquals(0, alphaVal);
        assertEquals(1, betaVal);
        assertEquals(2, gammaVal);
    }

    @Test
    public void testSortedIndexValues() {
        // Build expects keys sorted by unsigned byte order. For ASCII,
        // this matches lexicographic order. Values are the sorted index.
        // Use a LinkedHashSet to verify the build sorts internally.
        Set<String> words = new LinkedHashSet<>();
        words.add("cherry");
        words.add("apple");
        words.add("banana");
        Trie trie = new Trie(words);

        // Sorted order: apple(0), banana(1), cherry(2)
        assertEquals(0, trie.exactMatchSearch("apple".getBytes(StandardCharsets.UTF_8)).value());
        assertEquals(1, trie.exactMatchSearch("banana".getBytes(StandardCharsets.UTF_8)).value());
        assertEquals(2, trie.exactMatchSearch("cherry".getBytes(StandardCharsets.UTF_8)).value());
    }

    @Test
    public void testTraverseFullKey() {
        Set<String> words = Set.of("cat", "car", "card");
        Trie trie = new Trie(words);

        byte[] key = "cat".getBytes(StandardCharsets.UTF_8);
        Trie.TraverseResult result = trie.traverse(key, 0, 0);
        assertTrue(result.value() >= 0, "Expected to find 'cat' via traverse");
        assertEquals(key.length, result.queryPos());
    }

    @Test
    public void testTraverseNotFound() {
        Set<String> words = Set.of("cat", "car");
        Trie trie = new Trie(words);

        byte[] key = "caz".getBytes(StandardCharsets.UTF_8);
        Trie.TraverseResult result = trie.traverse(key, 0, 0);
        assertEquals(-2, result.value());
    }

    @Test
    public void testTraversePartialKey() {
        Set<String> words = Set.of("card");
        Trie trie = new Trie(words);

        // Traverse only "car" which is a prefix of "card" but not a key itself
        byte[] key = "car".getBytes(StandardCharsets.UTF_8);
        Trie.TraverseResult result = trie.traverse(key, 0, 0);
        assertEquals(-1, result.value(), "Prefix should return -1 (not an accept state)");
    }

    @Test
    public void testTraverseIncrementally() {
        Set<String> words = Set.of("abc");
        Trie trie = new Trie(words);

        byte[] full = "abc".getBytes(StandardCharsets.UTF_8);

        // Traverse "ab" first
        byte[] prefix = "ab".getBytes(StandardCharsets.UTF_8);
        Trie.TraverseResult partial = trie.traverse(prefix, 0, 0);
        assertEquals(-1, partial.value());

        // Continue from the saved node position with "c"
        byte[] suffix = "c".getBytes(StandardCharsets.UTF_8);
        Trie.TraverseResult complete = trie.traverse(suffix, partial.nodePos(), 0);
        assertTrue(complete.value() >= 0, "Completing traversal should find the key");

        // The value should match a direct full traversal
        Trie.TraverseResult direct = trie.traverse(full, 0, 0);
        assertEquals(direct.value(), complete.value());
    }

    @Test
    public void testManyKeys() {
        Set<String> words = new TreeSet<>();
        for (char c1 = 'a'; c1 <= 'z'; c1++) {
            for (char c2 = 'a'; c2 <= 'z'; c2++) {
                words.add("" + c1 + c2);
            }
        }
        // 676 two-letter keys
        assertEquals(676, words.size());
        Trie trie = new Trie(words);

        for (String word : words) {
            byte[] key = word.getBytes(StandardCharsets.UTF_8);
            Trie.TrieResult result = trie.exactMatchSearch(key);
            assertTrue(result.value() >= 0, "Expected to find '" + word + "'");
            assertEquals(key.length, result.length());
        }

        // Verify a missing key
        assertEquals(-1, trie.exactMatchSearch("a".getBytes(StandardCharsets.UTF_8)).value());
        assertEquals(-1, trie.exactMatchSearch("abc".getBytes(StandardCharsets.UTF_8)).value());
    }

    @Test
    public void testSingleCharacterKeys() {
        Set<String> words = new TreeSet<>();
        for (char c = 'a'; c <= 'z'; c++) {
            words.add(String.valueOf(c));
        }
        Trie trie = new Trie(words);

        for (String word : words) {
            byte[] key = word.getBytes(StandardCharsets.UTF_8);
            Trie.TrieResult result = trie.exactMatchSearch(key);
            assertTrue(result.value() >= 0, "Expected to find '" + word + "'");
        }
    }

    @Test
    public void testHasLeaf() {
        // Bit 8 set
        assertTrue(Trie.hasLeaf(1 << 8));
        assertTrue(Trie.hasLeaf(0xFFFFFFFF));
        // Bit 8 not set
        assertFalse(Trie.hasLeaf(0));
        assertFalse(Trie.hasLeaf(0xFF));
        assertFalse(Trie.hasLeaf(~(1 << 8)));
    }

    @Test
    public void testValue() {
        // value() masks off the MSB
        assertEquals(0, Trie.value(0));
        assertEquals(42, Trie.value(42 | (1 << 31)));
        assertEquals(Integer.MAX_VALUE, Trie.value(0xFFFFFFFF));
    }

    @Test
    public void testLabel() {
        // label() returns lower 8 bits + MSB
        assertEquals(0, Trie.label(0));
        assertEquals(0x41, Trie.label(0x41));
        assertEquals(0xFF, Trie.label(0xFF));
        // MSB is preserved
        assertEquals((1 << 31) | 0x42, Trie.label((1 << 31) | 0x42));
        // Middle bits are masked out
        assertEquals(0x41, Trie.label(0x7FFFFF41));
    }

    @Test
    public void testOffset() {
        // offset = (input >> 10) << ((input & (1 << 9)) >> 6)
        // When bit 9 is 0: shift amount is 0, so offset = input >> 10
        assertEquals(1, Trie.offset(1 << 10));
        assertEquals(0, Trie.offset(0));
        // When bit 9 is 1: shift amount is (1<<9)>>6 = 8, so offset = (input >> 10) << 8
        int withBit9 = (1 << 10) | (1 << 9);
        assertEquals(1 << 8, Trie.offset(withBit9));
    }

    @Test
    public void testDuplicateKeysInSet() {
        // A Set naturally deduplicates, but verify the trie handles it
        Set<String> words = new TreeSet<>();
        words.add("hello");
        words.add("world");
        Trie trie = new Trie(words);

        assertTrue(trie.exactMatchSearch("hello".getBytes(StandardCharsets.UTF_8)).value() >= 0);
        assertTrue(trie.exactMatchSearch("world".getBytes(StandardCharsets.UTF_8)).value() >= 0);
    }

    @Test
    public void testLongerKeys() {
        Set<String> words = Set.of(
                "the quick brown fox",
                "the quick brown dog",
                "the slow red fox",
                "a lazy cat"
        );
        Trie trie = new Trie(words);

        for (String word : words) {
            byte[] key = word.getBytes(StandardCharsets.UTF_8);
            Trie.TrieResult result = trie.exactMatchSearch(key);
            assertTrue(result.value() >= 0, "Expected to find '" + word + "'");
            assertEquals(key.length, result.length());
        }

        assertEquals(-1,
                trie.exactMatchSearch("the quick brown".getBytes(StandardCharsets.UTF_8)).value());
    }

    @Test
    public void testKeyIsSubstringOfAnother() {
        Set<String> words = Set.of("be", "bee", "been", "beer");
        Trie trie = new Trie(words);

        for (String word : words) {
            byte[] key = word.getBytes(StandardCharsets.UTF_8);
            Trie.TrieResult result = trie.exactMatchSearch(key);
            assertTrue(result.value() >= 0, "Expected to find '" + word + "'");
        }

        // "b" is not in the set
        assertEquals(-1, trie.exactMatchSearch("b".getBytes(StandardCharsets.UTF_8)).value());
        // "beep" is not in the set
        assertEquals(-1, trie.exactMatchSearch("beep".getBytes(StandardCharsets.UTF_8)).value());
    }

    @Test
    public void testEmptySet() {
        Trie trie = new Trie(Set.of());

        // Nothing should be found
        assertEquals(-1, trie.exactMatchSearch("a".getBytes(StandardCharsets.UTF_8)).value());
    }

    @Test
    public void testTraverseConsistentWithExactMatch() {
        Set<String> words = Set.of("foo", "bar", "baz", "foobar");
        Trie trie = new Trie(words);

        for (String word : words) {
            byte[] key = word.getBytes(StandardCharsets.UTF_8);
            Trie.TrieResult exact = trie.exactMatchSearch(key);
            Trie.TraverseResult traversed = trie.traverse(key, 0, 0);

            assertEquals(exact.value(), traversed.value(),
                    "exactMatchSearch and traverse should agree on value for '" + word + "'");
        }
    }
}
