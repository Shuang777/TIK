import numpy
import struct

def read_utterance (ark):
    ## Read utterance ID
    uid = b''
    c = ark.read(1)
    if not c:
        return None, None
    while c != b' ':
        uid += c
        c = ark.read(1)
    ## Read feature matrix
    header = struct.unpack('<xcccc', ark.read(5))
    m, rows = struct.unpack('<bi', ark.read(5))
    n, cols = struct.unpack('<bi', ark.read(5))
    feat = numpy.frombuffer(ark.read(rows * cols * 4), dtype=numpy.float32)
    return uid, feat.reshape((rows,cols))


def write_utterance (uid, feat, ark, encoding):
    feat = numpy.asarray (feat, dtype=numpy.float32)
    m,n = feat.shape
    ## Write header
    ark.write (struct.pack('<%ds'%(len(uid)), uid))
    ark.write (struct.pack('<cxcccc',' ','B',
                'F','M',' '))
    ark.write (struct.pack('<bi', 4, m))
    ark.write (struct.pack('<bi', 4, n))
    ## Write feature matrix
    ark.write (feat)

