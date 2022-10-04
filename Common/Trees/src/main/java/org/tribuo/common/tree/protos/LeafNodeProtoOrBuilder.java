// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-tree.proto

package org.tribuo.common.tree.protos;

public interface LeafNodeProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.common.tree.LeafNodeProto)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>int32 parent_idx = 1;</code>
   * @return The parentIdx.
   */
  int getParentIdx();

  /**
   * <code>int32 cur_idx = 2;</code>
   * @return The curIdx.
   */
  int getCurIdx();

  /**
   * <code>double impurity = 3;</code>
   * @return The impurity.
   */
  double getImpurity();

  /**
   * <code>.tribuo.core.OutputProto output = 4;</code>
   * @return Whether the output field is set.
   */
  boolean hasOutput();
  /**
   * <code>.tribuo.core.OutputProto output = 4;</code>
   * @return The output.
   */
  org.tribuo.protos.core.OutputProto getOutput();
  /**
   * <code>.tribuo.core.OutputProto output = 4;</code>
   */
  org.tribuo.protos.core.OutputProtoOrBuilder getOutputOrBuilder();

  /**
   * <code>map&lt;string, .tribuo.core.OutputProto&gt; score = 5;</code>
   */
  int getScoreCount();
  /**
   * <code>map&lt;string, .tribuo.core.OutputProto&gt; score = 5;</code>
   */
  boolean containsScore(
      java.lang.String key);
  /**
   * Use {@link #getScoreMap()} instead.
   */
  @java.lang.Deprecated
  java.util.Map<java.lang.String, org.tribuo.protos.core.OutputProto>
  getScore();
  /**
   * <code>map&lt;string, .tribuo.core.OutputProto&gt; score = 5;</code>
   */
  java.util.Map<java.lang.String, org.tribuo.protos.core.OutputProto>
  getScoreMap();
  /**
   * <code>map&lt;string, .tribuo.core.OutputProto&gt; score = 5;</code>
   */

  org.tribuo.protos.core.OutputProto getScoreOrDefault(
      java.lang.String key,
      org.tribuo.protos.core.OutputProto defaultValue);
  /**
   * <code>map&lt;string, .tribuo.core.OutputProto&gt; score = 5;</code>
   */

  org.tribuo.protos.core.OutputProto getScoreOrThrow(
      java.lang.String key);

  /**
   * <code>bool generates_probabilities = 6;</code>
   * @return The generatesProbabilities.
   */
  boolean getGeneratesProbabilities();
}