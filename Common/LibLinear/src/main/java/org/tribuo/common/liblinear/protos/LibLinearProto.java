// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-liblinear.proto

package org.tribuo.common.liblinear.protos;

/**
 * <pre>
 *Liblinear model data proto
 * </pre>
 *
 * Protobuf type {@code tribuo.common.liblinear.LibLinearProto}
 */
public final class LibLinearProto extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tribuo.common.liblinear.LibLinearProto)
    LibLinearProtoOrBuilder {
private static final long serialVersionUID = 0L;
  // Use LibLinearProto.newBuilder() to construct.
  private LibLinearProto(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private LibLinearProto() {
    label_ = emptyIntList();
    solverType_ = "";
    w_ = emptyDoubleList();
  }

  @java.lang.Override
  @SuppressWarnings({"unused"})
  protected java.lang.Object newInstance(
      UnusedPrivateParameter unused) {
    return new LibLinearProto();
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private LibLinearProto(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    this();
    if (extensionRegistry == null) {
      throw new java.lang.NullPointerException();
    }
    int mutable_bitField0_ = 0;
    com.google.protobuf.UnknownFieldSet.Builder unknownFields =
        com.google.protobuf.UnknownFieldSet.newBuilder();
    try {
      boolean done = false;
      while (!done) {
        int tag = input.readTag();
        switch (tag) {
          case 0:
            done = true;
            break;
          case 9: {

            bias_ = input.readDouble();
            break;
          }
          case 16: {
            if (!((mutable_bitField0_ & 0x00000001) != 0)) {
              label_ = newIntList();
              mutable_bitField0_ |= 0x00000001;
            }
            label_.addInt(input.readInt32());
            break;
          }
          case 18: {
            int length = input.readRawVarint32();
            int limit = input.pushLimit(length);
            if (!((mutable_bitField0_ & 0x00000001) != 0) && input.getBytesUntilLimit() > 0) {
              label_ = newIntList();
              mutable_bitField0_ |= 0x00000001;
            }
            while (input.getBytesUntilLimit() > 0) {
              label_.addInt(input.readInt32());
            }
            input.popLimit(limit);
            break;
          }
          case 24: {

            nrClass_ = input.readInt32();
            break;
          }
          case 32: {

            nrFeature_ = input.readInt32();
            break;
          }
          case 42: {
            java.lang.String s = input.readStringRequireUtf8();

            solverType_ = s;
            break;
          }
          case 49: {
            if (!((mutable_bitField0_ & 0x00000002) != 0)) {
              w_ = newDoubleList();
              mutable_bitField0_ |= 0x00000002;
            }
            w_.addDouble(input.readDouble());
            break;
          }
          case 50: {
            int length = input.readRawVarint32();
            int limit = input.pushLimit(length);
            if (!((mutable_bitField0_ & 0x00000002) != 0) && input.getBytesUntilLimit() > 0) {
              w_ = newDoubleList();
              mutable_bitField0_ |= 0x00000002;
            }
            while (input.getBytesUntilLimit() > 0) {
              w_.addDouble(input.readDouble());
            }
            input.popLimit(limit);
            break;
          }
          case 57: {

            rho_ = input.readDouble();
            break;
          }
          default: {
            if (!parseUnknownField(
                input, unknownFields, extensionRegistry, tag)) {
              done = true;
            }
            break;
          }
        }
      }
    } catch (com.google.protobuf.InvalidProtocolBufferException e) {
      throw e.setUnfinishedMessage(this);
    } catch (java.io.IOException e) {
      throw new com.google.protobuf.InvalidProtocolBufferException(
          e).setUnfinishedMessage(this);
    } finally {
      if (((mutable_bitField0_ & 0x00000001) != 0)) {
        label_.makeImmutable(); // C
      }
      if (((mutable_bitField0_ & 0x00000002) != 0)) {
        w_.makeImmutable(); // C
      }
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.tribuo.common.liblinear.protos.TribuoLiblinear.internal_static_tribuo_common_liblinear_LibLinearProto_descriptor;
  }

  @java.lang.Override
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tribuo.common.liblinear.protos.TribuoLiblinear.internal_static_tribuo_common_liblinear_LibLinearProto_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tribuo.common.liblinear.protos.LibLinearProto.class, org.tribuo.common.liblinear.protos.LibLinearProto.Builder.class);
  }

  public static final int BIAS_FIELD_NUMBER = 1;
  private double bias_;
  /**
   * <code>double bias = 1;</code>
   * @return The bias.
   */
  @java.lang.Override
  public double getBias() {
    return bias_;
  }

  public static final int LABEL_FIELD_NUMBER = 2;
  private com.google.protobuf.Internal.IntList label_;
  /**
   * <code>repeated int32 label = 2;</code>
   * @return A list containing the label.
   */
  @java.lang.Override
  public java.util.List<java.lang.Integer>
      getLabelList() {
    return label_;
  }
  /**
   * <code>repeated int32 label = 2;</code>
   * @return The count of label.
   */
  public int getLabelCount() {
    return label_.size();
  }
  /**
   * <code>repeated int32 label = 2;</code>
   * @param index The index of the element to return.
   * @return The label at the given index.
   */
  public int getLabel(int index) {
    return label_.getInt(index);
  }
  private int labelMemoizedSerializedSize = -1;

  public static final int NR_CLASS_FIELD_NUMBER = 3;
  private int nrClass_;
  /**
   * <code>int32 nr_class = 3;</code>
   * @return The nrClass.
   */
  @java.lang.Override
  public int getNrClass() {
    return nrClass_;
  }

  public static final int NR_FEATURE_FIELD_NUMBER = 4;
  private int nrFeature_;
  /**
   * <code>int32 nr_feature = 4;</code>
   * @return The nrFeature.
   */
  @java.lang.Override
  public int getNrFeature() {
    return nrFeature_;
  }

  public static final int SOLVER_TYPE_FIELD_NUMBER = 5;
  private volatile java.lang.Object solverType_;
  /**
   * <code>string solver_type = 5;</code>
   * @return The solverType.
   */
  @java.lang.Override
  public java.lang.String getSolverType() {
    java.lang.Object ref = solverType_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      solverType_ = s;
      return s;
    }
  }
  /**
   * <code>string solver_type = 5;</code>
   * @return The bytes for solverType.
   */
  @java.lang.Override
  public com.google.protobuf.ByteString
      getSolverTypeBytes() {
    java.lang.Object ref = solverType_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      solverType_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  public static final int W_FIELD_NUMBER = 6;
  private com.google.protobuf.Internal.DoubleList w_;
  /**
   * <code>repeated double w = 6;</code>
   * @return A list containing the w.
   */
  @java.lang.Override
  public java.util.List<java.lang.Double>
      getWList() {
    return w_;
  }
  /**
   * <code>repeated double w = 6;</code>
   * @return The count of w.
   */
  public int getWCount() {
    return w_.size();
  }
  /**
   * <code>repeated double w = 6;</code>
   * @param index The index of the element to return.
   * @return The w at the given index.
   */
  public double getW(int index) {
    return w_.getDouble(index);
  }
  private int wMemoizedSerializedSize = -1;

  public static final int RHO_FIELD_NUMBER = 7;
  private double rho_;
  /**
   * <code>double rho = 7;</code>
   * @return The rho.
   */
  @java.lang.Override
  public double getRho() {
    return rho_;
  }

  private byte memoizedIsInitialized = -1;
  @java.lang.Override
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    memoizedIsInitialized = 1;
    return true;
  }

  @java.lang.Override
  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    getSerializedSize();
    if (java.lang.Double.doubleToRawLongBits(bias_) != 0) {
      output.writeDouble(1, bias_);
    }
    if (getLabelList().size() > 0) {
      output.writeUInt32NoTag(18);
      output.writeUInt32NoTag(labelMemoizedSerializedSize);
    }
    for (int i = 0; i < label_.size(); i++) {
      output.writeInt32NoTag(label_.getInt(i));
    }
    if (nrClass_ != 0) {
      output.writeInt32(3, nrClass_);
    }
    if (nrFeature_ != 0) {
      output.writeInt32(4, nrFeature_);
    }
    if (!com.google.protobuf.GeneratedMessageV3.isStringEmpty(solverType_)) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 5, solverType_);
    }
    if (getWList().size() > 0) {
      output.writeUInt32NoTag(50);
      output.writeUInt32NoTag(wMemoizedSerializedSize);
    }
    for (int i = 0; i < w_.size(); i++) {
      output.writeDoubleNoTag(w_.getDouble(i));
    }
    if (java.lang.Double.doubleToRawLongBits(rho_) != 0) {
      output.writeDouble(7, rho_);
    }
    unknownFields.writeTo(output);
  }

  @java.lang.Override
  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (java.lang.Double.doubleToRawLongBits(bias_) != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeDoubleSize(1, bias_);
    }
    {
      int dataSize = 0;
      for (int i = 0; i < label_.size(); i++) {
        dataSize += com.google.protobuf.CodedOutputStream
          .computeInt32SizeNoTag(label_.getInt(i));
      }
      size += dataSize;
      if (!getLabelList().isEmpty()) {
        size += 1;
        size += com.google.protobuf.CodedOutputStream
            .computeInt32SizeNoTag(dataSize);
      }
      labelMemoizedSerializedSize = dataSize;
    }
    if (nrClass_ != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt32Size(3, nrClass_);
    }
    if (nrFeature_ != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt32Size(4, nrFeature_);
    }
    if (!com.google.protobuf.GeneratedMessageV3.isStringEmpty(solverType_)) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(5, solverType_);
    }
    {
      int dataSize = 0;
      dataSize = 8 * getWList().size();
      size += dataSize;
      if (!getWList().isEmpty()) {
        size += 1;
        size += com.google.protobuf.CodedOutputStream
            .computeInt32SizeNoTag(dataSize);
      }
      wMemoizedSerializedSize = dataSize;
    }
    if (java.lang.Double.doubleToRawLongBits(rho_) != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeDoubleSize(7, rho_);
    }
    size += unknownFields.getSerializedSize();
    memoizedSize = size;
    return size;
  }

  @java.lang.Override
  public boolean equals(final java.lang.Object obj) {
    if (obj == this) {
     return true;
    }
    if (!(obj instanceof org.tribuo.common.liblinear.protos.LibLinearProto)) {
      return super.equals(obj);
    }
    org.tribuo.common.liblinear.protos.LibLinearProto other = (org.tribuo.common.liblinear.protos.LibLinearProto) obj;

    if (java.lang.Double.doubleToLongBits(getBias())
        != java.lang.Double.doubleToLongBits(
            other.getBias())) return false;
    if (!getLabelList()
        .equals(other.getLabelList())) return false;
    if (getNrClass()
        != other.getNrClass()) return false;
    if (getNrFeature()
        != other.getNrFeature()) return false;
    if (!getSolverType()
        .equals(other.getSolverType())) return false;
    if (!getWList()
        .equals(other.getWList())) return false;
    if (java.lang.Double.doubleToLongBits(getRho())
        != java.lang.Double.doubleToLongBits(
            other.getRho())) return false;
    if (!unknownFields.equals(other.unknownFields)) return false;
    return true;
  }

  @java.lang.Override
  public int hashCode() {
    if (memoizedHashCode != 0) {
      return memoizedHashCode;
    }
    int hash = 41;
    hash = (19 * hash) + getDescriptor().hashCode();
    hash = (37 * hash) + BIAS_FIELD_NUMBER;
    hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
        java.lang.Double.doubleToLongBits(getBias()));
    if (getLabelCount() > 0) {
      hash = (37 * hash) + LABEL_FIELD_NUMBER;
      hash = (53 * hash) + getLabelList().hashCode();
    }
    hash = (37 * hash) + NR_CLASS_FIELD_NUMBER;
    hash = (53 * hash) + getNrClass();
    hash = (37 * hash) + NR_FEATURE_FIELD_NUMBER;
    hash = (53 * hash) + getNrFeature();
    hash = (37 * hash) + SOLVER_TYPE_FIELD_NUMBER;
    hash = (53 * hash) + getSolverType().hashCode();
    if (getWCount() > 0) {
      hash = (37 * hash) + W_FIELD_NUMBER;
      hash = (53 * hash) + getWList().hashCode();
    }
    hash = (37 * hash) + RHO_FIELD_NUMBER;
    hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
        java.lang.Double.doubleToLongBits(getRho()));
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tribuo.common.liblinear.protos.LibLinearProto parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.common.liblinear.protos.LibLinearProto parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.common.liblinear.protos.LibLinearProto parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.common.liblinear.protos.LibLinearProto parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.common.liblinear.protos.LibLinearProto parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.common.liblinear.protos.LibLinearProto parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.common.liblinear.protos.LibLinearProto parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.common.liblinear.protos.LibLinearProto parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tribuo.common.liblinear.protos.LibLinearProto parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.tribuo.common.liblinear.protos.LibLinearProto parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tribuo.common.liblinear.protos.LibLinearProto parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.common.liblinear.protos.LibLinearProto parseFrom(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }

  @java.lang.Override
  public Builder newBuilderForType() { return newBuilder(); }
  public static Builder newBuilder() {
    return DEFAULT_INSTANCE.toBuilder();
  }
  public static Builder newBuilder(org.tribuo.common.liblinear.protos.LibLinearProto prototype) {
    return DEFAULT_INSTANCE.toBuilder().mergeFrom(prototype);
  }
  @java.lang.Override
  public Builder toBuilder() {
    return this == DEFAULT_INSTANCE
        ? new Builder() : new Builder().mergeFrom(this);
  }

  @java.lang.Override
  protected Builder newBuilderForType(
      com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
    Builder builder = new Builder(parent);
    return builder;
  }
  /**
   * <pre>
   *Liblinear model data proto
   * </pre>
   *
   * Protobuf type {@code tribuo.common.liblinear.LibLinearProto}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tribuo.common.liblinear.LibLinearProto)
      org.tribuo.common.liblinear.protos.LibLinearProtoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tribuo.common.liblinear.protos.TribuoLiblinear.internal_static_tribuo_common_liblinear_LibLinearProto_descriptor;
    }

    @java.lang.Override
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tribuo.common.liblinear.protos.TribuoLiblinear.internal_static_tribuo_common_liblinear_LibLinearProto_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tribuo.common.liblinear.protos.LibLinearProto.class, org.tribuo.common.liblinear.protos.LibLinearProto.Builder.class);
    }

    // Construct using org.tribuo.common.liblinear.protos.LibLinearProto.newBuilder()
    private Builder() {
      maybeForceBuilderInitialization();
    }

    private Builder(
        com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
      super(parent);
      maybeForceBuilderInitialization();
    }
    private void maybeForceBuilderInitialization() {
      if (com.google.protobuf.GeneratedMessageV3
              .alwaysUseFieldBuilders) {
      }
    }
    @java.lang.Override
    public Builder clear() {
      super.clear();
      bias_ = 0D;

      label_ = emptyIntList();
      bitField0_ = (bitField0_ & ~0x00000001);
      nrClass_ = 0;

      nrFeature_ = 0;

      solverType_ = "";

      w_ = emptyDoubleList();
      bitField0_ = (bitField0_ & ~0x00000002);
      rho_ = 0D;

      return this;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tribuo.common.liblinear.protos.TribuoLiblinear.internal_static_tribuo_common_liblinear_LibLinearProto_descriptor;
    }

    @java.lang.Override
    public org.tribuo.common.liblinear.protos.LibLinearProto getDefaultInstanceForType() {
      return org.tribuo.common.liblinear.protos.LibLinearProto.getDefaultInstance();
    }

    @java.lang.Override
    public org.tribuo.common.liblinear.protos.LibLinearProto build() {
      org.tribuo.common.liblinear.protos.LibLinearProto result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    @java.lang.Override
    public org.tribuo.common.liblinear.protos.LibLinearProto buildPartial() {
      org.tribuo.common.liblinear.protos.LibLinearProto result = new org.tribuo.common.liblinear.protos.LibLinearProto(this);
      int from_bitField0_ = bitField0_;
      result.bias_ = bias_;
      if (((bitField0_ & 0x00000001) != 0)) {
        label_.makeImmutable();
        bitField0_ = (bitField0_ & ~0x00000001);
      }
      result.label_ = label_;
      result.nrClass_ = nrClass_;
      result.nrFeature_ = nrFeature_;
      result.solverType_ = solverType_;
      if (((bitField0_ & 0x00000002) != 0)) {
        w_.makeImmutable();
        bitField0_ = (bitField0_ & ~0x00000002);
      }
      result.w_ = w_;
      result.rho_ = rho_;
      onBuilt();
      return result;
    }

    @java.lang.Override
    public Builder clone() {
      return super.clone();
    }
    @java.lang.Override
    public Builder setField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        java.lang.Object value) {
      return super.setField(field, value);
    }
    @java.lang.Override
    public Builder clearField(
        com.google.protobuf.Descriptors.FieldDescriptor field) {
      return super.clearField(field);
    }
    @java.lang.Override
    public Builder clearOneof(
        com.google.protobuf.Descriptors.OneofDescriptor oneof) {
      return super.clearOneof(oneof);
    }
    @java.lang.Override
    public Builder setRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        int index, java.lang.Object value) {
      return super.setRepeatedField(field, index, value);
    }
    @java.lang.Override
    public Builder addRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        java.lang.Object value) {
      return super.addRepeatedField(field, value);
    }
    @java.lang.Override
    public Builder mergeFrom(com.google.protobuf.Message other) {
      if (other instanceof org.tribuo.common.liblinear.protos.LibLinearProto) {
        return mergeFrom((org.tribuo.common.liblinear.protos.LibLinearProto)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tribuo.common.liblinear.protos.LibLinearProto other) {
      if (other == org.tribuo.common.liblinear.protos.LibLinearProto.getDefaultInstance()) return this;
      if (other.getBias() != 0D) {
        setBias(other.getBias());
      }
      if (!other.label_.isEmpty()) {
        if (label_.isEmpty()) {
          label_ = other.label_;
          bitField0_ = (bitField0_ & ~0x00000001);
        } else {
          ensureLabelIsMutable();
          label_.addAll(other.label_);
        }
        onChanged();
      }
      if (other.getNrClass() != 0) {
        setNrClass(other.getNrClass());
      }
      if (other.getNrFeature() != 0) {
        setNrFeature(other.getNrFeature());
      }
      if (!other.getSolverType().isEmpty()) {
        solverType_ = other.solverType_;
        onChanged();
      }
      if (!other.w_.isEmpty()) {
        if (w_.isEmpty()) {
          w_ = other.w_;
          bitField0_ = (bitField0_ & ~0x00000002);
        } else {
          ensureWIsMutable();
          w_.addAll(other.w_);
        }
        onChanged();
      }
      if (other.getRho() != 0D) {
        setRho(other.getRho());
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    @java.lang.Override
    public final boolean isInitialized() {
      return true;
    }

    @java.lang.Override
    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.tribuo.common.liblinear.protos.LibLinearProto parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.tribuo.common.liblinear.protos.LibLinearProto) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private double bias_ ;
    /**
     * <code>double bias = 1;</code>
     * @return The bias.
     */
    @java.lang.Override
    public double getBias() {
      return bias_;
    }
    /**
     * <code>double bias = 1;</code>
     * @param value The bias to set.
     * @return This builder for chaining.
     */
    public Builder setBias(double value) {
      
      bias_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>double bias = 1;</code>
     * @return This builder for chaining.
     */
    public Builder clearBias() {
      
      bias_ = 0D;
      onChanged();
      return this;
    }

    private com.google.protobuf.Internal.IntList label_ = emptyIntList();
    private void ensureLabelIsMutable() {
      if (!((bitField0_ & 0x00000001) != 0)) {
        label_ = mutableCopy(label_);
        bitField0_ |= 0x00000001;
       }
    }
    /**
     * <code>repeated int32 label = 2;</code>
     * @return A list containing the label.
     */
    public java.util.List<java.lang.Integer>
        getLabelList() {
      return ((bitField0_ & 0x00000001) != 0) ?
               java.util.Collections.unmodifiableList(label_) : label_;
    }
    /**
     * <code>repeated int32 label = 2;</code>
     * @return The count of label.
     */
    public int getLabelCount() {
      return label_.size();
    }
    /**
     * <code>repeated int32 label = 2;</code>
     * @param index The index of the element to return.
     * @return The label at the given index.
     */
    public int getLabel(int index) {
      return label_.getInt(index);
    }
    /**
     * <code>repeated int32 label = 2;</code>
     * @param index The index to set the value at.
     * @param value The label to set.
     * @return This builder for chaining.
     */
    public Builder setLabel(
        int index, int value) {
      ensureLabelIsMutable();
      label_.setInt(index, value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated int32 label = 2;</code>
     * @param value The label to add.
     * @return This builder for chaining.
     */
    public Builder addLabel(int value) {
      ensureLabelIsMutable();
      label_.addInt(value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated int32 label = 2;</code>
     * @param values The label to add.
     * @return This builder for chaining.
     */
    public Builder addAllLabel(
        java.lang.Iterable<? extends java.lang.Integer> values) {
      ensureLabelIsMutable();
      com.google.protobuf.AbstractMessageLite.Builder.addAll(
          values, label_);
      onChanged();
      return this;
    }
    /**
     * <code>repeated int32 label = 2;</code>
     * @return This builder for chaining.
     */
    public Builder clearLabel() {
      label_ = emptyIntList();
      bitField0_ = (bitField0_ & ~0x00000001);
      onChanged();
      return this;
    }

    private int nrClass_ ;
    /**
     * <code>int32 nr_class = 3;</code>
     * @return The nrClass.
     */
    @java.lang.Override
    public int getNrClass() {
      return nrClass_;
    }
    /**
     * <code>int32 nr_class = 3;</code>
     * @param value The nrClass to set.
     * @return This builder for chaining.
     */
    public Builder setNrClass(int value) {
      
      nrClass_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>int32 nr_class = 3;</code>
     * @return This builder for chaining.
     */
    public Builder clearNrClass() {
      
      nrClass_ = 0;
      onChanged();
      return this;
    }

    private int nrFeature_ ;
    /**
     * <code>int32 nr_feature = 4;</code>
     * @return The nrFeature.
     */
    @java.lang.Override
    public int getNrFeature() {
      return nrFeature_;
    }
    /**
     * <code>int32 nr_feature = 4;</code>
     * @param value The nrFeature to set.
     * @return This builder for chaining.
     */
    public Builder setNrFeature(int value) {
      
      nrFeature_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>int32 nr_feature = 4;</code>
     * @return This builder for chaining.
     */
    public Builder clearNrFeature() {
      
      nrFeature_ = 0;
      onChanged();
      return this;
    }

    private java.lang.Object solverType_ = "";
    /**
     * <code>string solver_type = 5;</code>
     * @return The solverType.
     */
    public java.lang.String getSolverType() {
      java.lang.Object ref = solverType_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        solverType_ = s;
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <code>string solver_type = 5;</code>
     * @return The bytes for solverType.
     */
    public com.google.protobuf.ByteString
        getSolverTypeBytes() {
      java.lang.Object ref = solverType_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        solverType_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <code>string solver_type = 5;</code>
     * @param value The solverType to set.
     * @return This builder for chaining.
     */
    public Builder setSolverType(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  
      solverType_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>string solver_type = 5;</code>
     * @return This builder for chaining.
     */
    public Builder clearSolverType() {
      
      solverType_ = getDefaultInstance().getSolverType();
      onChanged();
      return this;
    }
    /**
     * <code>string solver_type = 5;</code>
     * @param value The bytes for solverType to set.
     * @return This builder for chaining.
     */
    public Builder setSolverTypeBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  checkByteStringIsUtf8(value);
      
      solverType_ = value;
      onChanged();
      return this;
    }

    private com.google.protobuf.Internal.DoubleList w_ = emptyDoubleList();
    private void ensureWIsMutable() {
      if (!((bitField0_ & 0x00000002) != 0)) {
        w_ = mutableCopy(w_);
        bitField0_ |= 0x00000002;
       }
    }
    /**
     * <code>repeated double w = 6;</code>
     * @return A list containing the w.
     */
    public java.util.List<java.lang.Double>
        getWList() {
      return ((bitField0_ & 0x00000002) != 0) ?
               java.util.Collections.unmodifiableList(w_) : w_;
    }
    /**
     * <code>repeated double w = 6;</code>
     * @return The count of w.
     */
    public int getWCount() {
      return w_.size();
    }
    /**
     * <code>repeated double w = 6;</code>
     * @param index The index of the element to return.
     * @return The w at the given index.
     */
    public double getW(int index) {
      return w_.getDouble(index);
    }
    /**
     * <code>repeated double w = 6;</code>
     * @param index The index to set the value at.
     * @param value The w to set.
     * @return This builder for chaining.
     */
    public Builder setW(
        int index, double value) {
      ensureWIsMutable();
      w_.setDouble(index, value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated double w = 6;</code>
     * @param value The w to add.
     * @return This builder for chaining.
     */
    public Builder addW(double value) {
      ensureWIsMutable();
      w_.addDouble(value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated double w = 6;</code>
     * @param values The w to add.
     * @return This builder for chaining.
     */
    public Builder addAllW(
        java.lang.Iterable<? extends java.lang.Double> values) {
      ensureWIsMutable();
      com.google.protobuf.AbstractMessageLite.Builder.addAll(
          values, w_);
      onChanged();
      return this;
    }
    /**
     * <code>repeated double w = 6;</code>
     * @return This builder for chaining.
     */
    public Builder clearW() {
      w_ = emptyDoubleList();
      bitField0_ = (bitField0_ & ~0x00000002);
      onChanged();
      return this;
    }

    private double rho_ ;
    /**
     * <code>double rho = 7;</code>
     * @return The rho.
     */
    @java.lang.Override
    public double getRho() {
      return rho_;
    }
    /**
     * <code>double rho = 7;</code>
     * @param value The rho to set.
     * @return This builder for chaining.
     */
    public Builder setRho(double value) {
      
      rho_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>double rho = 7;</code>
     * @return This builder for chaining.
     */
    public Builder clearRho() {
      
      rho_ = 0D;
      onChanged();
      return this;
    }
    @java.lang.Override
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFields(unknownFields);
    }

    @java.lang.Override
    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:tribuo.common.liblinear.LibLinearProto)
  }

  // @@protoc_insertion_point(class_scope:tribuo.common.liblinear.LibLinearProto)
  private static final org.tribuo.common.liblinear.protos.LibLinearProto DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tribuo.common.liblinear.protos.LibLinearProto();
  }

  public static org.tribuo.common.liblinear.protos.LibLinearProto getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<LibLinearProto>
      PARSER = new com.google.protobuf.AbstractParser<LibLinearProto>() {
    @java.lang.Override
    public LibLinearProto parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return new LibLinearProto(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<LibLinearProto> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<LibLinearProto> getParserForType() {
    return PARSER;
  }

  @java.lang.Override
  public org.tribuo.common.liblinear.protos.LibLinearProto getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
