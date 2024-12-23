// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tribuo-core-impl.proto

// Protobuf Java Version: 3.25.5
package org.tribuo.protos.core;

/**
 * <pre>
 *
 *IndependentSequenceModel proto
 * </pre>
 *
 * Protobuf type {@code tribuo.core.IndependentSequenceModelProto}
 */
public final class IndependentSequenceModelProto extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:tribuo.core.IndependentSequenceModelProto)
    IndependentSequenceModelProtoOrBuilder {
private static final long serialVersionUID = 0L;
  // Use IndependentSequenceModelProto.newBuilder() to construct.
  private IndependentSequenceModelProto(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private IndependentSequenceModelProto() {
  }

  @java.lang.Override
  @SuppressWarnings({"unused"})
  protected java.lang.Object newInstance(
      UnusedPrivateParameter unused) {
    return new IndependentSequenceModelProto();
  }

  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.tribuo.protos.core.TribuoCoreImpl.internal_static_tribuo_core_IndependentSequenceModelProto_descriptor;
  }

  @java.lang.Override
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.tribuo.protos.core.TribuoCoreImpl.internal_static_tribuo_core_IndependentSequenceModelProto_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.tribuo.protos.core.IndependentSequenceModelProto.class, org.tribuo.protos.core.IndependentSequenceModelProto.Builder.class);
  }

  private int bitField0_;
  public static final int METADATA_FIELD_NUMBER = 1;
  private org.tribuo.protos.core.ModelDataProto metadata_;
  /**
   * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
   * @return Whether the metadata field is set.
   */
  @java.lang.Override
  public boolean hasMetadata() {
    return ((bitField0_ & 0x00000001) != 0);
  }
  /**
   * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
   * @return The metadata.
   */
  @java.lang.Override
  public org.tribuo.protos.core.ModelDataProto getMetadata() {
    return metadata_ == null ? org.tribuo.protos.core.ModelDataProto.getDefaultInstance() : metadata_;
  }
  /**
   * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
   */
  @java.lang.Override
  public org.tribuo.protos.core.ModelDataProtoOrBuilder getMetadataOrBuilder() {
    return metadata_ == null ? org.tribuo.protos.core.ModelDataProto.getDefaultInstance() : metadata_;
  }

  public static final int MODEL_FIELD_NUMBER = 2;
  private org.tribuo.protos.core.ModelProto model_;
  /**
   * <code>.tribuo.core.ModelProto model = 2;</code>
   * @return Whether the model field is set.
   */
  @java.lang.Override
  public boolean hasModel() {
    return ((bitField0_ & 0x00000002) != 0);
  }
  /**
   * <code>.tribuo.core.ModelProto model = 2;</code>
   * @return The model.
   */
  @java.lang.Override
  public org.tribuo.protos.core.ModelProto getModel() {
    return model_ == null ? org.tribuo.protos.core.ModelProto.getDefaultInstance() : model_;
  }
  /**
   * <code>.tribuo.core.ModelProto model = 2;</code>
   */
  @java.lang.Override
  public org.tribuo.protos.core.ModelProtoOrBuilder getModelOrBuilder() {
    return model_ == null ? org.tribuo.protos.core.ModelProto.getDefaultInstance() : model_;
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
    if (((bitField0_ & 0x00000001) != 0)) {
      output.writeMessage(1, getMetadata());
    }
    if (((bitField0_ & 0x00000002) != 0)) {
      output.writeMessage(2, getModel());
    }
    getUnknownFields().writeTo(output);
  }

  @java.lang.Override
  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) != 0)) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(1, getMetadata());
    }
    if (((bitField0_ & 0x00000002) != 0)) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(2, getModel());
    }
    size += getUnknownFields().getSerializedSize();
    memoizedSize = size;
    return size;
  }

  @java.lang.Override
  public boolean equals(final java.lang.Object obj) {
    if (obj == this) {
     return true;
    }
    if (!(obj instanceof org.tribuo.protos.core.IndependentSequenceModelProto)) {
      return super.equals(obj);
    }
    org.tribuo.protos.core.IndependentSequenceModelProto other = (org.tribuo.protos.core.IndependentSequenceModelProto) obj;

    if (hasMetadata() != other.hasMetadata()) return false;
    if (hasMetadata()) {
      if (!getMetadata()
          .equals(other.getMetadata())) return false;
    }
    if (hasModel() != other.hasModel()) return false;
    if (hasModel()) {
      if (!getModel()
          .equals(other.getModel())) return false;
    }
    if (!getUnknownFields().equals(other.getUnknownFields())) return false;
    return true;
  }

  @java.lang.Override
  public int hashCode() {
    if (memoizedHashCode != 0) {
      return memoizedHashCode;
    }
    int hash = 41;
    hash = (19 * hash) + getDescriptor().hashCode();
    if (hasMetadata()) {
      hash = (37 * hash) + METADATA_FIELD_NUMBER;
      hash = (53 * hash) + getMetadata().hashCode();
    }
    if (hasModel()) {
      hash = (37 * hash) + MODEL_FIELD_NUMBER;
      hash = (53 * hash) + getModel().hashCode();
    }
    hash = (29 * hash) + getUnknownFields().hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.tribuo.protos.core.IndependentSequenceModelProto parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.protos.core.IndependentSequenceModelProto parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.protos.core.IndependentSequenceModelProto parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.protos.core.IndependentSequenceModelProto parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.protos.core.IndependentSequenceModelProto parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.tribuo.protos.core.IndependentSequenceModelProto parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.tribuo.protos.core.IndependentSequenceModelProto parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.protos.core.IndependentSequenceModelProto parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }

  public static org.tribuo.protos.core.IndependentSequenceModelProto parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }

  public static org.tribuo.protos.core.IndependentSequenceModelProto parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.tribuo.protos.core.IndependentSequenceModelProto parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.tribuo.protos.core.IndependentSequenceModelProto parseFrom(
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
  public static Builder newBuilder(org.tribuo.protos.core.IndependentSequenceModelProto prototype) {
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
   *
   *IndependentSequenceModel proto
   * </pre>
   *
   * Protobuf type {@code tribuo.core.IndependentSequenceModelProto}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:tribuo.core.IndependentSequenceModelProto)
      org.tribuo.protos.core.IndependentSequenceModelProtoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.tribuo.protos.core.TribuoCoreImpl.internal_static_tribuo_core_IndependentSequenceModelProto_descriptor;
    }

    @java.lang.Override
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.tribuo.protos.core.TribuoCoreImpl.internal_static_tribuo_core_IndependentSequenceModelProto_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.tribuo.protos.core.IndependentSequenceModelProto.class, org.tribuo.protos.core.IndependentSequenceModelProto.Builder.class);
    }

    // Construct using org.tribuo.protos.core.IndependentSequenceModelProto.newBuilder()
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
        getMetadataFieldBuilder();
        getModelFieldBuilder();
      }
    }
    @java.lang.Override
    public Builder clear() {
      super.clear();
      bitField0_ = 0;
      metadata_ = null;
      if (metadataBuilder_ != null) {
        metadataBuilder_.dispose();
        metadataBuilder_ = null;
      }
      model_ = null;
      if (modelBuilder_ != null) {
        modelBuilder_.dispose();
        modelBuilder_ = null;
      }
      return this;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.tribuo.protos.core.TribuoCoreImpl.internal_static_tribuo_core_IndependentSequenceModelProto_descriptor;
    }

    @java.lang.Override
    public org.tribuo.protos.core.IndependentSequenceModelProto getDefaultInstanceForType() {
      return org.tribuo.protos.core.IndependentSequenceModelProto.getDefaultInstance();
    }

    @java.lang.Override
    public org.tribuo.protos.core.IndependentSequenceModelProto build() {
      org.tribuo.protos.core.IndependentSequenceModelProto result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    @java.lang.Override
    public org.tribuo.protos.core.IndependentSequenceModelProto buildPartial() {
      org.tribuo.protos.core.IndependentSequenceModelProto result = new org.tribuo.protos.core.IndependentSequenceModelProto(this);
      if (bitField0_ != 0) { buildPartial0(result); }
      onBuilt();
      return result;
    }

    private void buildPartial0(org.tribuo.protos.core.IndependentSequenceModelProto result) {
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) != 0)) {
        result.metadata_ = metadataBuilder_ == null
            ? metadata_
            : metadataBuilder_.build();
        to_bitField0_ |= 0x00000001;
      }
      if (((from_bitField0_ & 0x00000002) != 0)) {
        result.model_ = modelBuilder_ == null
            ? model_
            : modelBuilder_.build();
        to_bitField0_ |= 0x00000002;
      }
      result.bitField0_ |= to_bitField0_;
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
      if (other instanceof org.tribuo.protos.core.IndependentSequenceModelProto) {
        return mergeFrom((org.tribuo.protos.core.IndependentSequenceModelProto)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.tribuo.protos.core.IndependentSequenceModelProto other) {
      if (other == org.tribuo.protos.core.IndependentSequenceModelProto.getDefaultInstance()) return this;
      if (other.hasMetadata()) {
        mergeMetadata(other.getMetadata());
      }
      if (other.hasModel()) {
        mergeModel(other.getModel());
      }
      this.mergeUnknownFields(other.getUnknownFields());
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
      if (extensionRegistry == null) {
        throw new java.lang.NullPointerException();
      }
      try {
        boolean done = false;
        while (!done) {
          int tag = input.readTag();
          switch (tag) {
            case 0:
              done = true;
              break;
            case 10: {
              input.readMessage(
                  getMetadataFieldBuilder().getBuilder(),
                  extensionRegistry);
              bitField0_ |= 0x00000001;
              break;
            } // case 10
            case 18: {
              input.readMessage(
                  getModelFieldBuilder().getBuilder(),
                  extensionRegistry);
              bitField0_ |= 0x00000002;
              break;
            } // case 18
            default: {
              if (!super.parseUnknownField(input, extensionRegistry, tag)) {
                done = true; // was an endgroup tag
              }
              break;
            } // default:
          } // switch (tag)
        } // while (!done)
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        throw e.unwrapIOException();
      } finally {
        onChanged();
      } // finally
      return this;
    }
    private int bitField0_;

    private org.tribuo.protos.core.ModelDataProto metadata_;
    private com.google.protobuf.SingleFieldBuilderV3<
        org.tribuo.protos.core.ModelDataProto, org.tribuo.protos.core.ModelDataProto.Builder, org.tribuo.protos.core.ModelDataProtoOrBuilder> metadataBuilder_;
    /**
     * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
     * @return Whether the metadata field is set.
     */
    public boolean hasMetadata() {
      return ((bitField0_ & 0x00000001) != 0);
    }
    /**
     * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
     * @return The metadata.
     */
    public org.tribuo.protos.core.ModelDataProto getMetadata() {
      if (metadataBuilder_ == null) {
        return metadata_ == null ? org.tribuo.protos.core.ModelDataProto.getDefaultInstance() : metadata_;
      } else {
        return metadataBuilder_.getMessage();
      }
    }
    /**
     * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
     */
    public Builder setMetadata(org.tribuo.protos.core.ModelDataProto value) {
      if (metadataBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        metadata_ = value;
      } else {
        metadataBuilder_.setMessage(value);
      }
      bitField0_ |= 0x00000001;
      onChanged();
      return this;
    }
    /**
     * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
     */
    public Builder setMetadata(
        org.tribuo.protos.core.ModelDataProto.Builder builderForValue) {
      if (metadataBuilder_ == null) {
        metadata_ = builderForValue.build();
      } else {
        metadataBuilder_.setMessage(builderForValue.build());
      }
      bitField0_ |= 0x00000001;
      onChanged();
      return this;
    }
    /**
     * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
     */
    public Builder mergeMetadata(org.tribuo.protos.core.ModelDataProto value) {
      if (metadataBuilder_ == null) {
        if (((bitField0_ & 0x00000001) != 0) &&
          metadata_ != null &&
          metadata_ != org.tribuo.protos.core.ModelDataProto.getDefaultInstance()) {
          getMetadataBuilder().mergeFrom(value);
        } else {
          metadata_ = value;
        }
      } else {
        metadataBuilder_.mergeFrom(value);
      }
      if (metadata_ != null) {
        bitField0_ |= 0x00000001;
        onChanged();
      }
      return this;
    }
    /**
     * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
     */
    public Builder clearMetadata() {
      bitField0_ = (bitField0_ & ~0x00000001);
      metadata_ = null;
      if (metadataBuilder_ != null) {
        metadataBuilder_.dispose();
        metadataBuilder_ = null;
      }
      onChanged();
      return this;
    }
    /**
     * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
     */
    public org.tribuo.protos.core.ModelDataProto.Builder getMetadataBuilder() {
      bitField0_ |= 0x00000001;
      onChanged();
      return getMetadataFieldBuilder().getBuilder();
    }
    /**
     * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
     */
    public org.tribuo.protos.core.ModelDataProtoOrBuilder getMetadataOrBuilder() {
      if (metadataBuilder_ != null) {
        return metadataBuilder_.getMessageOrBuilder();
      } else {
        return metadata_ == null ?
            org.tribuo.protos.core.ModelDataProto.getDefaultInstance() : metadata_;
      }
    }
    /**
     * <code>.tribuo.core.ModelDataProto metadata = 1;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        org.tribuo.protos.core.ModelDataProto, org.tribuo.protos.core.ModelDataProto.Builder, org.tribuo.protos.core.ModelDataProtoOrBuilder> 
        getMetadataFieldBuilder() {
      if (metadataBuilder_ == null) {
        metadataBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            org.tribuo.protos.core.ModelDataProto, org.tribuo.protos.core.ModelDataProto.Builder, org.tribuo.protos.core.ModelDataProtoOrBuilder>(
                getMetadata(),
                getParentForChildren(),
                isClean());
        metadata_ = null;
      }
      return metadataBuilder_;
    }

    private org.tribuo.protos.core.ModelProto model_;
    private com.google.protobuf.SingleFieldBuilderV3<
        org.tribuo.protos.core.ModelProto, org.tribuo.protos.core.ModelProto.Builder, org.tribuo.protos.core.ModelProtoOrBuilder> modelBuilder_;
    /**
     * <code>.tribuo.core.ModelProto model = 2;</code>
     * @return Whether the model field is set.
     */
    public boolean hasModel() {
      return ((bitField0_ & 0x00000002) != 0);
    }
    /**
     * <code>.tribuo.core.ModelProto model = 2;</code>
     * @return The model.
     */
    public org.tribuo.protos.core.ModelProto getModel() {
      if (modelBuilder_ == null) {
        return model_ == null ? org.tribuo.protos.core.ModelProto.getDefaultInstance() : model_;
      } else {
        return modelBuilder_.getMessage();
      }
    }
    /**
     * <code>.tribuo.core.ModelProto model = 2;</code>
     */
    public Builder setModel(org.tribuo.protos.core.ModelProto value) {
      if (modelBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        model_ = value;
      } else {
        modelBuilder_.setMessage(value);
      }
      bitField0_ |= 0x00000002;
      onChanged();
      return this;
    }
    /**
     * <code>.tribuo.core.ModelProto model = 2;</code>
     */
    public Builder setModel(
        org.tribuo.protos.core.ModelProto.Builder builderForValue) {
      if (modelBuilder_ == null) {
        model_ = builderForValue.build();
      } else {
        modelBuilder_.setMessage(builderForValue.build());
      }
      bitField0_ |= 0x00000002;
      onChanged();
      return this;
    }
    /**
     * <code>.tribuo.core.ModelProto model = 2;</code>
     */
    public Builder mergeModel(org.tribuo.protos.core.ModelProto value) {
      if (modelBuilder_ == null) {
        if (((bitField0_ & 0x00000002) != 0) &&
          model_ != null &&
          model_ != org.tribuo.protos.core.ModelProto.getDefaultInstance()) {
          getModelBuilder().mergeFrom(value);
        } else {
          model_ = value;
        }
      } else {
        modelBuilder_.mergeFrom(value);
      }
      if (model_ != null) {
        bitField0_ |= 0x00000002;
        onChanged();
      }
      return this;
    }
    /**
     * <code>.tribuo.core.ModelProto model = 2;</code>
     */
    public Builder clearModel() {
      bitField0_ = (bitField0_ & ~0x00000002);
      model_ = null;
      if (modelBuilder_ != null) {
        modelBuilder_.dispose();
        modelBuilder_ = null;
      }
      onChanged();
      return this;
    }
    /**
     * <code>.tribuo.core.ModelProto model = 2;</code>
     */
    public org.tribuo.protos.core.ModelProto.Builder getModelBuilder() {
      bitField0_ |= 0x00000002;
      onChanged();
      return getModelFieldBuilder().getBuilder();
    }
    /**
     * <code>.tribuo.core.ModelProto model = 2;</code>
     */
    public org.tribuo.protos.core.ModelProtoOrBuilder getModelOrBuilder() {
      if (modelBuilder_ != null) {
        return modelBuilder_.getMessageOrBuilder();
      } else {
        return model_ == null ?
            org.tribuo.protos.core.ModelProto.getDefaultInstance() : model_;
      }
    }
    /**
     * <code>.tribuo.core.ModelProto model = 2;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        org.tribuo.protos.core.ModelProto, org.tribuo.protos.core.ModelProto.Builder, org.tribuo.protos.core.ModelProtoOrBuilder> 
        getModelFieldBuilder() {
      if (modelBuilder_ == null) {
        modelBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            org.tribuo.protos.core.ModelProto, org.tribuo.protos.core.ModelProto.Builder, org.tribuo.protos.core.ModelProtoOrBuilder>(
                getModel(),
                getParentForChildren(),
                isClean());
        model_ = null;
      }
      return modelBuilder_;
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


    // @@protoc_insertion_point(builder_scope:tribuo.core.IndependentSequenceModelProto)
  }

  // @@protoc_insertion_point(class_scope:tribuo.core.IndependentSequenceModelProto)
  private static final org.tribuo.protos.core.IndependentSequenceModelProto DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.tribuo.protos.core.IndependentSequenceModelProto();
  }

  public static org.tribuo.protos.core.IndependentSequenceModelProto getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<IndependentSequenceModelProto>
      PARSER = new com.google.protobuf.AbstractParser<IndependentSequenceModelProto>() {
    @java.lang.Override
    public IndependentSequenceModelProto parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      Builder builder = newBuilder();
      try {
        builder.mergeFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        throw e.setUnfinishedMessage(builder.buildPartial());
      } catch (com.google.protobuf.UninitializedMessageException e) {
        throw e.asInvalidProtocolBufferException().setUnfinishedMessage(builder.buildPartial());
      } catch (java.io.IOException e) {
        throw new com.google.protobuf.InvalidProtocolBufferException(e)
            .setUnfinishedMessage(builder.buildPartial());
      }
      return builder.buildPartial();
    }
  };

  public static com.google.protobuf.Parser<IndependentSequenceModelProto> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<IndependentSequenceModelProto> getParserForType() {
    return PARSER;
  }

  @java.lang.Override
  public org.tribuo.protos.core.IndependentSequenceModelProto getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
