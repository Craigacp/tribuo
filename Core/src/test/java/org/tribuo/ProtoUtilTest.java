package org.tribuo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;
import org.tribuo.hash.HashCodeHasher;
import org.tribuo.hash.HashedFeatureMap;
import org.tribuo.hash.Hasher;
import org.tribuo.hash.MessageDigestHasher;
import org.tribuo.hash.ModHashCodeHasher;
import org.tribuo.protos.core.CategoricalIDInfoProto;
import org.tribuo.protos.core.CategoricalInfoProto;
import org.tribuo.protos.core.FeatureDomainProto;
import org.tribuo.protos.core.HashedFeatureMapProto;
import org.tribuo.protos.core.HasherProto;
import org.tribuo.protos.core.MessageDigestHasherProto;
import org.tribuo.protos.core.ModHashCodeHasherProto;
import org.tribuo.protos.core.RealIDInfoProto;
import org.tribuo.protos.core.RealInfoProto;
import org.tribuo.protos.core.VariableInfoProto;
import org.tribuo.protos.ProtoUtil;

public class ProtoUtilTest {

    @Test
    void testHashedFeatureMap() throws Exception {
        MutableFeatureMap mfm = new MutableFeatureMap(); 
        
        mfm.add("goldrat", 1.618033988749);
        mfm.add("e", Math.E);
        mfm.add("pi", Math.PI);
        HashedFeatureMap hfm = HashedFeatureMap.generateHashedFeatureMap(mfm, new MessageDigestHasher("SHA-512", "abcdefghi"));
        FeatureDomainProto fdp = hfm.serialize();
        assertEquals(0, fdp.getVersion());
        assertEquals("org.tribuo.hash.HashedFeatureMap", fdp.getClassName());
        HashedFeatureMapProto hfmp = fdp.getSerializedData().unpack(HashedFeatureMapProto.class);
        HasherProto hasherProto = hfmp.getHasher();
        assertEquals(0, hasherProto.getVersion());
        assertEquals("org.tribuo.hash.MessageDigestHasher", hasherProto.getClassName());
        MessageDigestHasherProto mdhp = hasherProto.getSerializedData().unpack(MessageDigestHasherProto.class);
        assertEquals("SHA-512", mdhp.getHashType());
        
        HashedFeatureMap hfmDeserialized = ProtoUtil.deserialize(fdp);
        hfmDeserialized.setSalt("abcdefghi");
        assertEquals(hfm, hfmDeserialized);
    }
    
    @Test
    void testSerializeModHashCodeHasherOld() throws Exception {
        ModHashCodeHasher hasher = new ModHashCodeHasher(200, "abcdefghi");
        HasherProto hasherProto = hasher.serialize();
        assertEquals(0, hasherProto.getVersion());
        assertEquals("org.tribuo.hash.ModHashCodeHasher", hasherProto.getClassName());
        
        ModHashCodeHasher hasherD = ProtoUtil.deserialize(hasherProto);
        hasherD.setSalt("abcdefghi");
        assertTrue(hasher.equals(hasherD));
        assertEquals(200, hasherProto.getSerializedData().unpack(ModHashCodeHasherProto.class).getDimension());
    }
    
    @Test
    void testSerializeModHashCodeHasher() throws Exception {
        ModHashCodeHasher hasher = new ModHashCodeHasher(200, "42");
        
        HasherProto hasherProto = ProtoUtil.serialize(hasher);
        assertEquals(0, hasherProto.getVersion());
        assertEquals("org.tribuo.hash.ModHashCodeHasher", hasherProto.getClassName());
        ModHashCodeHasherProto proto = hasherProto.getSerializedData().unpack(ModHashCodeHasherProto.class);
        assertEquals(200, proto.getDimension());

        Hasher hashr = new ModHashCodeHasher(200, "42");
        hasherProto = ProtoUtil.serialize(hashr);
        assertEquals(0, hasherProto.getVersion());
        assertEquals("org.tribuo.hash.ModHashCodeHasher", hasherProto.getClassName());
        proto = hasherProto.getSerializedData().unpack(ModHashCodeHasherProto.class);
        assertEquals(200, proto.getDimension());
    }
    
    @Test
    void testMessageDigestHasher() throws Exception {
        MessageDigestHasher hasher = new MessageDigestHasher("SHA-256", "42");
        HasherProto hasherProto = ProtoUtil.serialize(hasher);
        assertEquals(0, hasherProto.getVersion());
        assertEquals("org.tribuo.hash.MessageDigestHasher", hasherProto.getClassName());
        MessageDigestHasherProto proto = hasherProto.getSerializedData().unpack(MessageDigestHasherProto.class);
        assertEquals("SHA-256", proto.getHashType());
    }

    @Test
    void testHashCodeHasher() throws Exception {
        HashCodeHasher hasher = new HashCodeHasher("42");
        HasherProto hasherProto = ProtoUtil.serialize(hasher);
        assertEquals(0, hasherProto.getVersion());
        assertEquals("org.tribuo.hash.HashCodeHasher", hasherProto.getClassName());
        System.out.println(hasherProto.getSerializedData());
    }

    
    @Test
    void testRealIDInfo() throws Exception {
        VariableInfo info = new RealIDInfo("bob", 100, 1000.0, 0.0, 25.0, 125.0, 12345);
        VariableInfoProto infoProto = ProtoUtil.serialize(info);
        assertEquals(0, infoProto.getVersion());
        assertEquals("org.tribuo.RealIDInfo", infoProto.getClassName());
        RealIDInfoProto proto = infoProto.getSerializedData().unpack(RealIDInfoProto.class);
        assertEquals("bob", proto.getName());
        assertEquals(100, proto.getCount());
        assertEquals(1000.0, proto.getMax());
        assertEquals(0.0, proto.getMin());
        assertEquals(25.0, proto.getMean());
        assertEquals(125.0, proto.getSumSquares());
        assertEquals(12345, proto.getId());
    }
    
    @Test
    void testRealInfo() throws Exception {
        VariableInfo info = new RealInfo("bob", 100, 1000.0, 0.0, 25.0, 125.0);
        VariableInfoProto infoProto = ProtoUtil.serialize(info);
        assertEquals(0, infoProto.getVersion());
        assertEquals("org.tribuo.RealInfo", infoProto.getClassName());
        RealInfoProto proto = infoProto.getSerializedData().unpack(RealInfoProto.class);
        assertEquals("bob", proto.getName());
        assertEquals(100, proto.getCount());
        assertEquals(1000.0, proto.getMax());
        assertEquals(0.0, proto.getMin());
        assertEquals(25.0, proto.getMean());
        assertEquals(125.0, proto.getSumSquares());
    }

    
    @Test
    void testCategoricalInfo() throws Exception {
        CategoricalInfo info = new CategoricalInfo("cat");
        IntStream.range(0, 10).forEach(i -> {
            IntStream.range(0, i*2).forEach(j -> {
                info.observe(i);
            });
        });
        
        VariableInfoProto infoProto = info.serialize();
        assertEquals(0, infoProto.getVersion());
        assertEquals("org.tribuo.CategoricalInfo", infoProto.getClassName());
        CategoricalInfoProto proto = infoProto.getSerializedData().unpack(CategoricalInfoProto.class);
        assertEquals("cat", proto.getName());
        assertEquals(90, proto.getCount());
        assertEquals(0, proto.getObservedCount());
        assertEquals(Double.NaN, proto.getObservedValue());
        
        List<Double> keyList = proto.getKeyList();
        List<Long> valueList = proto.getValueList();

        assertEquals(9, keyList.size());
        assertEquals(9, valueList.size());
        
        Map<Double, Long> expectedCounts = new HashMap<>();
        IntStream.range(0, 10).forEach(i -> {
            long count = info.getObservationCount(i);
            expectedCounts.put((double)i, count);
        });
        
        for (int i=0; i<keyList.size(); i++) {
            assertEquals(expectedCounts.get(keyList.get(i)), valueList.get(i));
        }
    }

    @Test
    void testCategoricalInfo2() throws Exception {
        CategoricalInfo info = new CategoricalInfo("cat");
        IntStream.range(0, 10).forEach(i -> {
            info.observe(5);
        });
        
        VariableInfoProto infoProto = ProtoUtil.serialize(info);
        assertEquals(0, infoProto.getVersion());
        assertEquals("org.tribuo.CategoricalInfo", infoProto.getClassName());
        CategoricalInfoProto proto = infoProto.getSerializedData().unpack(CategoricalInfoProto.class);
        assertEquals("cat", proto.getName());
        assertEquals(10, proto.getCount());
        
        List<Double> keyList = proto.getKeyList();
        List<Long> valueList = proto.getValueList();

        assertEquals(0, keyList.size());
        assertEquals(0, valueList.size());
        assertEquals(5, proto.getObservedValue());
        assertEquals(10, proto.getObservedCount());
    }

    @Test
    void testCategoricalIdInfo() throws Exception {
        CategoricalInfo info = new CategoricalInfo("cat");
        IntStream.range(0, 10).forEach(i -> {
            IntStream.range(0, i*2).forEach(j -> {
                info.observe(i);
            });
        });

        CategoricalIDInfo idInfo = info.makeIDInfo(12345);
        
        VariableInfoProto infoProto = ProtoUtil.serialize(idInfo);
        assertEquals(0, infoProto.getVersion());
        assertEquals("org.tribuo.CategoricalIDInfo", infoProto.getClassName());
        CategoricalIDInfoProto proto = infoProto.getSerializedData().unpack(CategoricalIDInfoProto.class);
        assertEquals("cat", proto.getName());
        assertEquals(90, proto.getCount());
        assertEquals(12345, proto.getId());
        assertEquals(0, proto.getObservedCount());
        assertEquals(Double.NaN, proto.getObservedValue());
        
        List<Double> keyList = proto.getKeyList();
        List<Long> valueList = proto.getValueList();

        assertEquals(keyList.size(), valueList.size());
        
        Map<Double, Long> expectedCounts = new HashMap<>();
        IntStream.range(0, 10).forEach(i -> {
            long count = idInfo.getObservationCount(i);
            expectedCounts.put((double)i, count);
        });
        
        for (int i=0; i<keyList.size(); i++) {
            assertEquals(expectedCounts.get(keyList.get(i)), valueList.get(i));
        }
    }

    
    
}
