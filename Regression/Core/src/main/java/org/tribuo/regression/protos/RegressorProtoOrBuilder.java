// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-regression-core.proto

// Protobuf Java Version: 3.25.6
package org.tribuo.regression.protos;

public interface RegressorProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tribuo.regression.RegressorProto)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>repeated string name = 1;</code>
   * @return A list containing the name.
   */
  java.util.List<java.lang.String>
      getNameList();
  /**
   * <code>repeated string name = 1;</code>
   * @return The count of name.
   */
  int getNameCount();
  /**
   * <code>repeated string name = 1;</code>
   * @param index The index of the element to return.
   * @return The name at the given index.
   */
  java.lang.String getName(int index);
  /**
   * <code>repeated string name = 1;</code>
   * @param index The index of the value to return.
   * @return The bytes of the name at the given index.
   */
  com.google.protobuf.ByteString
      getNameBytes(int index);

  /**
   * <code>repeated double value = 2;</code>
   * @return A list containing the value.
   */
  java.util.List<java.lang.Double> getValueList();
  /**
   * <code>repeated double value = 2;</code>
   * @return The count of value.
   */
  int getValueCount();
  /**
   * <code>repeated double value = 2;</code>
   * @param index The index of the element to return.
   * @return The value at the given index.
   */
  double getValue(int index);

  /**
   * <code>repeated double variance = 3;</code>
   * @return A list containing the variance.
   */
  java.util.List<java.lang.Double> getVarianceList();
  /**
   * <code>repeated double variance = 3;</code>
   * @return The count of variance.
   */
  int getVarianceCount();
  /**
   * <code>repeated double variance = 3;</code>
   * @param index The index of the element to return.
   * @return The variance at the given index.
   */
  double getVariance(int index);
}
