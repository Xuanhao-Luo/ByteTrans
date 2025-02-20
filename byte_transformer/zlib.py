
import zlib

def compress_file(input_file_path, output_file_path):
    with open(input_file_path, 'rb') as input_file:
        data = input_file.read()
    compressed_data = zlib.compress(data, level=9)
    with open(output_file_path, 'wb') as output_file:
        output_file.write(compressed_data)
    print(f"File '{input_file_path}' compressed and saved as '{output_file_path}'")

def decompress_file(input_file_path, output_file_path):
    with open(input_file_path, 'rb') as input_file:
        compressed_data = input_file.read()
    decompressed_data = zlib.decompress(compressed_data)
    with open(output_file_path, 'wb') as output_file:
        output_file.write(decompressed_data)
    print(f"File '{input_file_path}' decompressed and saved as '{output_file_path}'")


#05M
compress_file('byte_transformer/test/rank_05M/rank20496.txt', 'byte_transformer/test/zlib_rank_05M/rank20496_05M.zlib')

#164M
# compress_file('byte_transformer/test/rank_5M_large/rank18192.txt', 'byte_transformer/test/zlib_rank_5M_large/rank18192.zlib')
# compress_file('byte_transformer/test/rank_5M_large/rank18448.txt', 'byte_transformer/test/zlib_rank_5M_large/rank18448.zlib')
# compress_file('byte_transformer/test/rank_5M_large/rank18704.txt', 'byte_transformer/test/zlib_rank_5M_large/rank18704.zlib')
# compress_file('byte_transformer/test/rank_5M_large/rank18960.txt', 'byte_transformer/test/zlib_rank_5M_large/rank18960.zlib')
# compress_file('byte_transformer/test/rank_5M_large/rank19216.txt', 'byte_transformer/test/zlib_rank_5M_large/rank19216.zlib')
# compress_file('byte_transformer/test/rank_5M_large/rank19472.txt', 'byte_transformer/test/zlib_rank_5M_large/rank19472.zlib')
# compress_file('byte_transformer/test/rank_5M_large/rank19728.txt', 'byte_transformer/test/zlib_rank_5M_large/rank19728.zlib')
# compress_file('byte_transformer/test/rank_5M_large/rank19984.txt', 'byte_transformer/test/zlib_rank_5M_large/rank19984.zlib')
# compress_file('byte_transformer/test/rank_5M_large/rank20240.txt', 'byte_transformer/test/zlib_rank_5M_large/rank20240.zlib')
# compress_file('byte_transformer/test/rank_5M_large/rank20496.txt', 'byte_transformer/test/zlib_rank_5M_large/rank20496.zlib')

#50M
# compress_file('byte_transformer/test/rank_50M/rank20496.txt', 'byte_transformer/test/zlib_rank_50M/rank20496_05M.zlib')


# decompress_file('input.txt.zlib', 'example_decompressed.txt')
