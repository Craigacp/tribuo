// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-core-impl.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.protos.core;

public interface SequenceExampleImplProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.core.SequenceExampleImplProto)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>repeated .tribuo.core.ExampleProto examples = 1;</code>
   */
  java.util.List<org.tribuo.protos.core.ExampleProto> 
      getExamplesList();
  /**
   * <code>repeated .tribuo.core.ExampleProto examples = 1;</code>
   */
  org.tribuo.protos.core.ExampleProto getExamples(int index);
  /**
   * <code>repeated .tribuo.core.ExampleProto examples = 1;</code>
   */
  int getExamplesCount();
  /**
   * <code>repeated .tribuo.core.ExampleProto examples = 1;</code>
   */
  java.util.List<? extends org.tribuo.protos.core.ExampleProtoOrBuilder> 
      getExamplesOrBuilderList();
  /**
   * <code>repeated .tribuo.core.ExampleProto examples = 1;</code>
   */
  org.tribuo.protos.core.ExampleProtoOrBuilder getExamplesOrBuilder(
      int index);

  /**
   * <code>float weight = 2;</code>
   * @return The weight.
   */
  float getWeight();
}
