import timeit
import torch
sum_512_768_ = torch.randn((512, 768,)).cuda()

maps_1_2_ = torch.randn((1, 2,)).cuda()

maps_1024_1_ = torch.randn((1024, 1,)).cuda()

matmul_grad2_2_12_512_64_1 = torch.randn((2, 12, 512, 64,)).cuda()
matmul_grad2_2_12_512_64_2 = torch.randn((2, 12, 512, 512,)).cuda()

matmul_grad1_2_12_512_512_1 = torch.randn((2, 12, 512, 512,)).cuda()
matmul_grad1_2_12_512_512_2 = torch.randn((2, 12, 64, 512,)).cuda()

sum_2048_768_ = torch.randn((2048, 768,)).cuda()

sum_16_12_512_512_ = torch.randn((16, 12, 512, 512,)).cuda()

matmul_2_768_1 = torch.randn((2, 768,)).cuda()
matmul_2_768_2 = torch.randn((16, 768, 1,)).cuda()

matmul_2_12_512_512_1 = torch.randn((2, 12, 512, 512,)).cuda()
matmul_2_12_512_512_2 = torch.randn((2, 12, 512, 64,)).cuda()

maps_8_512_768_1 = torch.randn((8, 512, 768,)).cuda()
maps_8_512_768_2 = torch.randn((8, 512, 768,)).cuda()

matmul_grad1_8_512_3072_1_1 = torch.randn((8, 512, 3072, 1,)).cuda()
matmul_grad1_8_512_3072_1_2 = torch.randn((8, 512, 768, 1,)).cuda()

matmul_grad2_2_768_1 = torch.randn((2, 768,)).cuda()
matmul_grad2_2_768_2 = torch.randn((16, 2, 1,)).cuda()

matmul_grad2_8_12_512_512_1 = torch.randn((8, 12, 512, 512,)).cuda()
matmul_grad2_8_12_512_512_2 = torch.randn((8, 12, 512, 64,)).cuda()

maps_4_512_768_1 = torch.randn((4, 512, 768,)).cuda()
maps_4_512_768_2 = torch.randn((4, 512, 768,)).cuda()

matmul_grad1_8_12_512_64_1 = torch.randn((8, 12, 512, 64,)).cuda()
matmul_grad1_8_12_512_64_2 = torch.randn((8, 12, 512, 64,)).cuda()

matmul_3072_768_1 = torch.randn((3072, 768,)).cuda()
matmul_3072_768_2 = torch.randn((16, 512, 768, 1,)).cuda()

maps_4_2_ = torch.randn((4, 2,)).cuda()

sumto_8_2_ = torch.randn((8, 2,)).cuda()

sumto_8192_768_ = torch.randn((8192, 768,)).cuda()

sum_2_12_512_512_ = torch.randn((2, 12, 512, 512,)).cuda()

maps_512_1_ = torch.randn((512, 1,)).cuda()

sumto_4_512_768_ = torch.randn((4, 512, 768,)).cuda()

matmul_grad1_16_12_512_512_1 = torch.randn((16, 12, 512, 512,)).cuda()
matmul_grad1_16_12_512_512_2 = torch.randn((16, 12, 64, 512,)).cuda()

maps_2_2_ = torch.randn((2, 2,)).cuda()

matmul_grad1_16_512_768_1_1 = torch.randn((16, 512, 768, 1,)).cuda()
matmul_grad1_16_512_768_1_2 = torch.randn((16, 512, 768, 1,)).cuda()

maps_2_1_1_512_ = torch.randn((2, 1, 1, 512,)).cuda()

sumto_8_512_768_ = torch.randn((8, 512, 768,)).cuda()

sum_1_12_512_512_ = torch.randn((1, 12, 512, 512,)).cuda()

maps_4_12_512_512_ = torch.randn((4, 12, 512, 512,)).cuda()

maps_8192_1_ = torch.randn((8192, 1,)).cuda()

matmul_grad2_1_12_512_64_1 = torch.randn((1, 12, 512, 64,)).cuda()
matmul_grad2_1_12_512_64_2 = torch.randn((1, 12, 512, 512,)).cuda()

matmul_grad1_1_2_1_1 = torch.randn((1, 2, 1,)).cuda()
matmul_grad1_1_2_1_2 = torch.randn((1, 768, 1,)).cuda()

matmul_16_12_512_64_1 = torch.randn((16, 12, 512, 64,)).cuda()
matmul_16_12_512_64_2 = torch.randn((16, 12, 64, 512,)).cuda()

maps_16_512_768_1 = torch.randn((16, 512, 768,)).cuda()
maps_16_512_768_2 = torch.randn((1, 512, 768,)).cuda()

matmul_grad1_4_512_3072_1_1 = torch.randn((4, 512, 3072, 1,)).cuda()
matmul_grad1_4_512_3072_1_2 = torch.randn((4, 512, 768, 1,)).cuda()

matmul_grad1_2_12_512_64_1 = torch.randn((2, 12, 512, 64,)).cuda()
matmul_grad1_2_12_512_64_2 = torch.randn((2, 12, 512, 64,)).cuda()

maps_8_512_3072_ = torch.randn((8, 512, 3072,)).cuda()

maps_512_768_1 = torch.randn((512, 768,)).cuda()
maps_512_768_2 = torch.randn((512, 768,)).cuda()

maps_1_512_3072_1 = torch.randn((1, 512, 3072,)).cuda()
maps_1_512_3072_2 = torch.randn((1, 512, 3072,)).cuda()

matmul_grad1_4_2_1_1 = torch.randn((4, 2, 1,)).cuda()
matmul_grad1_4_2_1_2 = torch.randn((4, 768, 1,)).cuda()

maps_2_12_512_512_ = torch.randn((2, 12, 512, 512,)).cuda()

sumto_1024_768_ = torch.randn((1024, 768,)).cuda()

maps_1_12_512_512_ = torch.randn((1, 12, 512, 512,)).cuda()

sumto_16_2_ = torch.randn((16, 2,)).cuda()

sumto_512_768_ = torch.randn((512, 768,)).cuda()

matmul_grad2_3072_768_1 = torch.randn((3072, 768,)).cuda()
matmul_grad2_3072_768_2 = torch.randn((16, 512, 3072, 1,)).cuda()

maps_8_12_512_512_ = torch.randn((8, 12, 512, 512,)).cuda()

matmul_grad2_768_768_1 = torch.randn((768, 768,)).cuda()
matmul_grad2_768_768_2 = torch.randn((16, 512, 768, 1,)).cuda()

maps_16_12_512_512_ = torch.randn((16, 12, 512, 512,)).cuda()

matmul_1_12_512_512_1 = torch.randn((1, 12, 512, 512,)).cuda()
matmul_1_12_512_512_2 = torch.randn((1, 12, 512, 64,)).cuda()

sum_8192_768_ = torch.randn((8192, 768,)).cuda()

matmul_4_12_512_512_1 = torch.randn((4, 12, 512, 512,)).cuda()
matmul_4_12_512_512_2 = torch.randn((4, 12, 512, 64,)).cuda()

matmul_grad1_4_12_512_512_1 = torch.randn((4, 12, 512, 512,)).cuda()
matmul_grad1_4_12_512_512_2 = torch.randn((4, 12, 64, 512,)).cuda()

maps_16_1_1_512_ = torch.randn((16, 1, 1, 512,)).cuda()

matmul_grad1_8_12_512_512_1 = torch.randn((8, 12, 512, 512,)).cuda()
matmul_grad1_8_12_512_512_2 = torch.randn((8, 12, 64, 512,)).cuda()

sumto_8_512_3072_ = torch.randn((8, 512, 3072,)).cuda()

matmul_768_768_1 = torch.randn((768, 768,)).cuda()
matmul_768_768_2 = torch.randn((16, 512, 768, 1,)).cuda()

maps_2048_1_ = torch.randn((2048, 1,)).cuda()

matmul_grad2_4_12_512_64_1 = torch.randn((4, 12, 512, 64,)).cuda()
matmul_grad2_4_12_512_64_2 = torch.randn((4, 12, 512, 512,)).cuda()

sumto_16_512_3072_ = torch.randn((16, 512, 3072,)).cuda()

matmul_grad1_1_512_768_1_1 = torch.randn((1, 512, 768, 1,)).cuda()
matmul_grad1_1_512_768_1_2 = torch.randn((1, 512, 768, 1,)).cuda()

matmul_grad1_1_12_512_64_1 = torch.randn((1, 12, 512, 64,)).cuda()
matmul_grad1_1_12_512_64_2 = torch.randn((1, 12, 512, 64,)).cuda()

sumto_1_2_ = torch.randn((1, 2,)).cuda()

matmul_2_12_512_64_1 = torch.randn((2, 12, 512, 64,)).cuda()
matmul_2_12_512_64_2 = torch.randn((2, 12, 64, 512,)).cuda()

matmul_4_12_512_64_1 = torch.randn((4, 12, 512, 64,)).cuda()
matmul_4_12_512_64_2 = torch.randn((4, 12, 64, 512,)).cuda()

matmul_768_3072_1 = torch.randn((768, 3072,)).cuda()
matmul_768_3072_2 = torch.randn((16, 512, 3072, 1,)).cuda()

sum_8_12_512_512_ = torch.randn((8, 12, 512, 512,)).cuda()

maps_4_512_3072_ = torch.randn((4, 512, 3072,)).cuda()

sumto_2048_768_ = torch.randn((2048, 768,)).cuda()

maps_1_512_768_1 = torch.randn((1, 512, 768,)).cuda()
maps_1_512_768_2 = torch.randn((1, 512, 768,)).cuda()

maps_1_1_1_512_ = torch.randn((1, 1, 1, 512,)).cuda()

maps_2_512_3072_1 = torch.randn((2, 512, 3072,)).cuda()
maps_2_512_3072_2 = torch.randn((2, 512, 3072,)).cuda()

sumto_4_2_ = torch.randn((4, 2,)).cuda()

maps_8_2_ = torch.randn((8, 2,)).cuda()

matmul_8_12_512_512_1 = torch.randn((8, 12, 512, 512,)).cuda()
matmul_8_12_512_512_2 = torch.randn((8, 12, 512, 64,)).cuda()

matmul_grad1_4_12_512_64_1 = torch.randn((4, 12, 512, 64,)).cuda()
matmul_grad1_4_12_512_64_2 = torch.randn((4, 12, 512, 64,)).cuda()

maps_4096_768_1 = torch.randn((4096, 768,)).cuda()
maps_4096_768_2 = torch.randn((1,)).cuda()

matmul_grad1_2_2_1_1 = torch.randn((2, 2, 1,)).cuda()
matmul_grad1_2_2_1_2 = torch.randn((2, 768, 1,)).cuda()

maps_4096_1_ = torch.randn((4096, 1,)).cuda()

matmul_grad1_16_12_512_64_1 = torch.randn((16, 12, 512, 64,)).cuda()
matmul_grad1_16_12_512_64_2 = torch.randn((16, 12, 512, 64,)).cuda()

maps_8_1_1_512_ = torch.randn((8, 1, 1, 512,)).cuda()

matmul_grad2_2_12_512_512_1 = torch.randn((2, 12, 512, 512,)).cuda()
matmul_grad2_2_12_512_512_2 = torch.randn((2, 12, 512, 64,)).cuda()

sumto_2_2_ = torch.randn((2, 2,)).cuda()

matmul_grad1_4_512_768_1_1 = torch.randn((4, 512, 768, 1,)).cuda()
matmul_grad1_4_512_768_1_2 = torch.randn((4, 512, 768, 1,)).cuda()

matmul_grad1_2_512_3072_1_1 = torch.randn((2, 512, 3072, 1,)).cuda()
matmul_grad1_2_512_3072_1_2 = torch.randn((2, 512, 768, 1,)).cuda()

sumto_2_512_3072_ = torch.randn((2, 512, 3072,)).cuda()

matmul_grad1_2_512_768_1_1 = torch.randn((2, 512, 768, 1,)).cuda()
matmul_grad1_2_512_768_1_2 = torch.randn((2, 512, 768, 1,)).cuda()

matmul_grad2_4_12_512_512_1 = torch.randn((4, 12, 512, 512,)).cuda()
matmul_grad2_4_12_512_512_2 = torch.randn((4, 12, 512, 64,)).cuda()

matmul_grad1_16_512_3072_1_1 = torch.randn((16, 512, 3072, 1,)).cuda()
matmul_grad1_16_512_3072_1_2 = torch.randn((16, 512, 768, 1,)).cuda()

maps_2048_768_1 = torch.randn((2048, 768,)).cuda()
maps_2048_768_2 = torch.randn((2048, 768,)).cuda()

matmul_grad1_1_512_3072_1_1 = torch.randn((1, 512, 3072, 1,)).cuda()
matmul_grad1_1_512_3072_1_2 = torch.randn((1, 512, 768, 1,)).cuda()

sum_4096_768_ = torch.randn((4096, 768,)).cuda()

sum_1024_768_ = torch.randn((1024, 768,)).cuda()

maps_1024_768_1 = torch.randn((1024, 768,)).cuda()
maps_1024_768_2 = torch.randn((1024, 768,)).cuda()

sumto_16_512_768_ = torch.randn((16, 512, 768,)).cuda()

matmul_grad2_8_12_512_64_1 = torch.randn((8, 12, 512, 64,)).cuda()
matmul_grad2_8_12_512_64_2 = torch.randn((8, 12, 512, 512,)).cuda()

sum_4_12_512_512_ = torch.randn((4, 12, 512, 512,)).cuda()

matmul_1_12_512_64_1 = torch.randn((1, 12, 512, 64,)).cuda()
matmul_1_12_512_64_2 = torch.randn((1, 12, 64, 512,)).cuda()

matmul_grad1_1_12_512_512_1 = torch.randn((1, 12, 512, 512,)).cuda()
matmul_grad1_1_12_512_512_2 = torch.randn((1, 12, 64, 512,)).cuda()

matmul_grad2_768_3072_1 = torch.randn((768, 3072,)).cuda()
matmul_grad2_768_3072_2 = torch.randn((16, 512, 768, 1,)).cuda()

matmul_16_12_512_512_1 = torch.randn((16, 12, 512, 512,)).cuda()
matmul_16_12_512_512_2 = torch.randn((16, 12, 512, 64,)).cuda()

maps_16_512_3072_1 = torch.randn((16, 512, 3072,)).cuda()
maps_16_512_3072_2 = torch.randn((16, 512, 3072,)).cuda()

maps_16_2_ = torch.randn((16, 2,)).cuda()

matmul_grad1_8_2_1_1 = torch.randn((8, 2, 1,)).cuda()
matmul_grad1_8_2_1_2 = torch.randn((8, 768, 1,)).cuda()

matmul_grad2_16_12_512_512_1 = torch.randn((16, 12, 512, 512,)).cuda()
matmul_grad2_16_12_512_512_2 = torch.randn((16, 12, 512, 64,)).cuda()

sumto_2_512_768_ = torch.randn((2, 512, 768,)).cuda()

maps_4_1_1_512_ = torch.randn((4, 1, 1, 512,)).cuda()

matmul_grad1_8_512_768_1_1 = torch.randn((8, 512, 768, 1,)).cuda()
matmul_grad1_8_512_768_1_2 = torch.randn((8, 512, 3072, 1,)).cuda()

maps_30522_768_1 = torch.randn((30522, 768,)).cuda()
maps_30522_768_2 = torch.randn((30522, 768,)).cuda()

matmul_grad2_1_12_512_512_1 = torch.randn((1, 12, 512, 512,)).cuda()
matmul_grad2_1_12_512_512_2 = torch.randn((1, 12, 512, 64,)).cuda()

sumto_1_512_768_ = torch.randn((1, 512, 768,)).cuda()

maps_2_512_768_1 = torch.randn((2, 512, 768,)).cuda()
maps_2_512_768_2 = torch.randn((2, 512, 768,)).cuda()

sumto_4_512_3072_ = torch.randn((4, 512, 3072,)).cuda()

matmul_grad2_16_12_512_64_1 = torch.randn((16, 12, 512, 64,)).cuda()
matmul_grad2_16_12_512_64_2 = torch.randn((16, 12, 512, 512,)).cuda()

maps_8192_768_1 = torch.randn((8192, 768,)).cuda()
maps_8192_768_2 = torch.randn((1,)).cuda()

matmul_8_12_512_64_1 = torch.randn((8, 12, 512, 64,)).cuda()
matmul_8_12_512_64_2 = torch.randn((8, 12, 64, 512,)).cuda()

matmul_grad1_16_2_1_1 = torch.randn((16, 2, 1,)).cuda()
matmul_grad1_16_2_1_2 = torch.randn((16, 768, 1,)).cuda()

sumto_1_512_3072_ = torch.randn((1, 512, 3072,)).cuda()

sumto_4096_768_ = torch.randn((4096, 768,)).cuda()


benchmark = {}
for i in range(2):
    t = timeit.timeit(lambda: torch.sum(sum_512_768_, dim=1), number=1)
    benchmark.setdefault('sum_512_768_', 0) 
    benchmark['sum_512_768_'] += t
    t = timeit.timeit(lambda: torch.nn.functional.relu(maps_1_2_), number=1)
    benchmark.setdefault('maps_1_2_', 0) 
    benchmark['maps_1_2_'] += t
    t = timeit.timeit(lambda: torch.nn.functional.relu(maps_1024_1_), number=1)
    benchmark.setdefault('maps_1024_1_', 0) 
    benchmark['maps_1024_1_'] += t
    t = timeit.timeit(lambda: torch.matmul(torch.transpose(matmul_grad2_2_12_512_64_1, -1, -2), matmul_grad2_2_12_512_64_2[0, :]), number=1)
    benchmark.setdefault('matmul_grad2_2_12_512_64_', 0) 
    benchmark['matmul_grad2_2_12_512_64_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_2_12_512_512_1[0, :], torch.transpose(matmul_grad1_2_12_512_512_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_2_12_512_512_', 0) 
    benchmark['matmul_grad1_2_12_512_512_'] += t
    t = timeit.timeit(lambda: torch.sum(sum_2048_768_, dim=1), number=1)
    benchmark.setdefault('sum_2048_768_', 0) 
    benchmark['sum_2048_768_'] += t
    t = timeit.timeit(lambda: torch.sum(sum_16_12_512_512_, dim=3), number=1)
    benchmark.setdefault('sum_16_12_512_512_', 0) 
    benchmark['sum_16_12_512_512_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_2_768_1, matmul_2_768_2), number=1)
    benchmark.setdefault('matmul_2_768_', 0) 
    benchmark['matmul_2_768_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_2_12_512_512_1, matmul_2_12_512_512_2), number=1)
    benchmark.setdefault('matmul_2_12_512_512_', 0) 
    benchmark['matmul_2_12_512_512_'] += t
    t = timeit.timeit(lambda: maps_8_512_768_1 + maps_8_512_768_2, number=1)
    benchmark.setdefault('maps_8_512_768_', 0) 
    benchmark['maps_8_512_768_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_8_512_3072_1_1[0, :], torch.transpose(matmul_grad1_8_512_3072_1_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_8_512_3072_1_', 0) 
    benchmark['matmul_grad1_8_512_3072_1_'] += t
    t = timeit.timeit(lambda: torch.matmul(torch.transpose(matmul_grad2_2_768_1, -1, -2), matmul_grad2_2_768_2[0, :]), number=1)
    benchmark.setdefault('matmul_grad2_2_768_', 0) 
    benchmark['matmul_grad2_2_768_'] += t
    t = timeit.timeit(lambda: torch.matmul(torch.transpose(matmul_grad2_8_12_512_512_1, -1, -2), matmul_grad2_8_12_512_512_2[0, :]), number=1)
    benchmark.setdefault('matmul_grad2_8_12_512_512_', 0) 
    benchmark['matmul_grad2_8_12_512_512_'] += t
    t = timeit.timeit(lambda: maps_4_512_768_1 + maps_4_512_768_2, number=1)
    benchmark.setdefault('maps_4_512_768_', 0) 
    benchmark['maps_4_512_768_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_8_12_512_64_1[0, :], torch.transpose(matmul_grad1_8_12_512_64_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_8_12_512_64_', 0) 
    benchmark['matmul_grad1_8_12_512_64_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_3072_768_1, matmul_3072_768_2), number=1)
    benchmark.setdefault('matmul_3072_768_', 0) 
    benchmark['matmul_3072_768_'] += t
    t = timeit.timeit(lambda: torch.nn.functional.relu(maps_4_2_), number=1)
    benchmark.setdefault('maps_4_2_', 0) 
    benchmark['maps_4_2_'] += t
    t = timeit.timeit(lambda: torch.sum(sumto_8_2_, dim=-1), number=1)
    benchmark.setdefault('sumto_8_2_', 0) 
    benchmark['sumto_8_2_'] += t
    t = timeit.timeit(lambda: torch.sum(sumto_8192_768_, dim=-1), number=1)
    benchmark.setdefault('sumto_8192_768_', 0) 
    benchmark['sumto_8192_768_'] += t
    t = timeit.timeit(lambda: torch.sum(sum_2_12_512_512_, dim=3), number=1)
    benchmark.setdefault('sum_2_12_512_512_', 0) 
    benchmark['sum_2_12_512_512_'] += t
    t = timeit.timeit(lambda: torch.nn.functional.relu(maps_512_1_), number=1)
    benchmark.setdefault('maps_512_1_', 0) 
    benchmark['maps_512_1_'] += t
    t = timeit.timeit(lambda: torch.sum(sumto_4_512_768_, dim=-1), number=1)
    benchmark.setdefault('sumto_4_512_768_', 0) 
    benchmark['sumto_4_512_768_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_16_12_512_512_1[0, :], torch.transpose(matmul_grad1_16_12_512_512_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_16_12_512_512_', 0) 
    benchmark['matmul_grad1_16_12_512_512_'] += t
    t = timeit.timeit(lambda: torch.nn.functional.relu(maps_2_2_), number=1)
    benchmark.setdefault('maps_2_2_', 0) 
    benchmark['maps_2_2_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_16_512_768_1_1[0, :], torch.transpose(matmul_grad1_16_512_768_1_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_16_512_768_1_', 0) 
    benchmark['matmul_grad1_16_512_768_1_'] += t
    t = timeit.timeit(lambda: torch.nn.functional.relu(maps_2_1_1_512_), number=1)
    benchmark.setdefault('maps_2_1_1_512_', 0) 
    benchmark['maps_2_1_1_512_'] += t
    t = timeit.timeit(lambda: torch.sum(sumto_8_512_768_, dim=-1), number=1)
    benchmark.setdefault('sumto_8_512_768_', 0) 
    benchmark['sumto_8_512_768_'] += t
    t = timeit.timeit(lambda: torch.sum(sum_1_12_512_512_, dim=3), number=1)
    benchmark.setdefault('sum_1_12_512_512_', 0) 
    benchmark['sum_1_12_512_512_'] += t
    t = timeit.timeit(lambda: torch.nn.functional.relu(maps_4_12_512_512_), number=1)
    benchmark.setdefault('maps_4_12_512_512_', 0) 
    benchmark['maps_4_12_512_512_'] += t
    t = timeit.timeit(lambda: torch.nn.functional.relu(maps_8192_1_), number=1)
    benchmark.setdefault('maps_8192_1_', 0) 
    benchmark['maps_8192_1_'] += t
    t = timeit.timeit(lambda: torch.matmul(torch.transpose(matmul_grad2_1_12_512_64_1, -1, -2), matmul_grad2_1_12_512_64_2[0, :]), number=1)
    benchmark.setdefault('matmul_grad2_1_12_512_64_', 0) 
    benchmark['matmul_grad2_1_12_512_64_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_1_2_1_1[0, :], torch.transpose(matmul_grad1_1_2_1_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_1_2_1_', 0) 
    benchmark['matmul_grad1_1_2_1_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_16_12_512_64_1, matmul_16_12_512_64_2), number=1)
    benchmark.setdefault('matmul_16_12_512_64_', 0) 
    benchmark['matmul_16_12_512_64_'] += t
    t = timeit.timeit(lambda: maps_16_512_768_1 + maps_16_512_768_2, number=1)
    benchmark.setdefault('maps_16_512_768_', 0) 
    benchmark['maps_16_512_768_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_4_512_3072_1_1[0, :], torch.transpose(matmul_grad1_4_512_3072_1_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_4_512_3072_1_', 0) 
    benchmark['matmul_grad1_4_512_3072_1_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_2_12_512_64_1[0, :], torch.transpose(matmul_grad1_2_12_512_64_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_2_12_512_64_', 0) 
    benchmark['matmul_grad1_2_12_512_64_'] += t
    t = timeit.timeit(lambda: torch.nn.functional.relu(maps_8_512_3072_), number=1)
    benchmark.setdefault('maps_8_512_3072_', 0) 
    benchmark['maps_8_512_3072_'] += t
    t = timeit.timeit(lambda: maps_512_768_1 + maps_512_768_2, number=1)
    benchmark.setdefault('maps_512_768_', 0) 
    benchmark['maps_512_768_'] += t
    t = timeit.timeit(lambda: maps_1_512_3072_1 + maps_1_512_3072_2, number=1)
    benchmark.setdefault('maps_1_512_3072_', 0) 
    benchmark['maps_1_512_3072_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_4_2_1_1[0, :], torch.transpose(matmul_grad1_4_2_1_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_4_2_1_', 0) 
    benchmark['matmul_grad1_4_2_1_'] += t
    t = timeit.timeit(lambda: torch.nn.functional.relu(maps_2_12_512_512_), number=1)
    benchmark.setdefault('maps_2_12_512_512_', 0) 
    benchmark['maps_2_12_512_512_'] += t
    t = timeit.timeit(lambda: torch.sum(sumto_1024_768_, dim=-1), number=1)
    benchmark.setdefault('sumto_1024_768_', 0) 
    benchmark['sumto_1024_768_'] += t
    t = timeit.timeit(lambda: torch.nn.functional.relu(maps_1_12_512_512_), number=1)
    benchmark.setdefault('maps_1_12_512_512_', 0) 
    benchmark['maps_1_12_512_512_'] += t
    t = timeit.timeit(lambda: torch.sum(sumto_16_2_, dim=-1), number=1)
    benchmark.setdefault('sumto_16_2_', 0) 
    benchmark['sumto_16_2_'] += t
    t = timeit.timeit(lambda: torch.sum(sumto_512_768_, dim=-1), number=1)
    benchmark.setdefault('sumto_512_768_', 0) 
    benchmark['sumto_512_768_'] += t
    t = timeit.timeit(lambda: torch.matmul(torch.transpose(matmul_grad2_3072_768_1, -1, -2), matmul_grad2_3072_768_2[0, :]), number=1)
    benchmark.setdefault('matmul_grad2_3072_768_', 0) 
    benchmark['matmul_grad2_3072_768_'] += t
    t = timeit.timeit(lambda: torch.nn.functional.relu(maps_8_12_512_512_), number=1)
    benchmark.setdefault('maps_8_12_512_512_', 0) 
    benchmark['maps_8_12_512_512_'] += t
    t = timeit.timeit(lambda: torch.matmul(torch.transpose(matmul_grad2_768_768_1, -1, -2), matmul_grad2_768_768_2[0, :]), number=1)
    benchmark.setdefault('matmul_grad2_768_768_', 0) 
    benchmark['matmul_grad2_768_768_'] += t
    t = timeit.timeit(lambda: torch.nn.functional.relu(maps_16_12_512_512_), number=1)
    benchmark.setdefault('maps_16_12_512_512_', 0) 
    benchmark['maps_16_12_512_512_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_1_12_512_512_1, matmul_1_12_512_512_2), number=1)
    benchmark.setdefault('matmul_1_12_512_512_', 0) 
    benchmark['matmul_1_12_512_512_'] += t
    t = timeit.timeit(lambda: torch.sum(sum_8192_768_, dim=1), number=1)
    benchmark.setdefault('sum_8192_768_', 0) 
    benchmark['sum_8192_768_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_4_12_512_512_1, matmul_4_12_512_512_2), number=1)
    benchmark.setdefault('matmul_4_12_512_512_', 0) 
    benchmark['matmul_4_12_512_512_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_4_12_512_512_1[0, :], torch.transpose(matmul_grad1_4_12_512_512_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_4_12_512_512_', 0) 
    benchmark['matmul_grad1_4_12_512_512_'] += t
    t = timeit.timeit(lambda: torch.nn.functional.relu(maps_16_1_1_512_), number=1)
    benchmark.setdefault('maps_16_1_1_512_', 0) 
    benchmark['maps_16_1_1_512_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_8_12_512_512_1[0, :], torch.transpose(matmul_grad1_8_12_512_512_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_8_12_512_512_', 0) 
    benchmark['matmul_grad1_8_12_512_512_'] += t
    t = timeit.timeit(lambda: torch.sum(sumto_8_512_3072_, dim=-1), number=1)
    benchmark.setdefault('sumto_8_512_3072_', 0) 
    benchmark['sumto_8_512_3072_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_768_768_1, matmul_768_768_2), number=1)
    benchmark.setdefault('matmul_768_768_', 0) 
    benchmark['matmul_768_768_'] += t
    t = timeit.timeit(lambda: torch.nn.functional.relu(maps_2048_1_), number=1)
    benchmark.setdefault('maps_2048_1_', 0) 
    benchmark['maps_2048_1_'] += t
    t = timeit.timeit(lambda: torch.matmul(torch.transpose(matmul_grad2_4_12_512_64_1, -1, -2), matmul_grad2_4_12_512_64_2[0, :]), number=1)
    benchmark.setdefault('matmul_grad2_4_12_512_64_', 0) 
    benchmark['matmul_grad2_4_12_512_64_'] += t
    t = timeit.timeit(lambda: torch.sum(sumto_16_512_3072_, dim=-1), number=1)
    benchmark.setdefault('sumto_16_512_3072_', 0) 
    benchmark['sumto_16_512_3072_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_1_512_768_1_1[0, :], torch.transpose(matmul_grad1_1_512_768_1_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_1_512_768_1_', 0) 
    benchmark['matmul_grad1_1_512_768_1_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_1_12_512_64_1[0, :], torch.transpose(matmul_grad1_1_12_512_64_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_1_12_512_64_', 0) 
    benchmark['matmul_grad1_1_12_512_64_'] += t
    t = timeit.timeit(lambda: torch.sum(sumto_1_2_, dim=-1), number=1)
    benchmark.setdefault('sumto_1_2_', 0) 
    benchmark['sumto_1_2_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_2_12_512_64_1, matmul_2_12_512_64_2), number=1)
    benchmark.setdefault('matmul_2_12_512_64_', 0) 
    benchmark['matmul_2_12_512_64_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_4_12_512_64_1, matmul_4_12_512_64_2), number=1)
    benchmark.setdefault('matmul_4_12_512_64_', 0) 
    benchmark['matmul_4_12_512_64_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_768_3072_1, matmul_768_3072_2), number=1)
    benchmark.setdefault('matmul_768_3072_', 0) 
    benchmark['matmul_768_3072_'] += t
    t = timeit.timeit(lambda: torch.sum(sum_8_12_512_512_, dim=3), number=1)
    benchmark.setdefault('sum_8_12_512_512_', 0) 
    benchmark['sum_8_12_512_512_'] += t
    t = timeit.timeit(lambda: torch.nn.functional.relu(maps_4_512_3072_), number=1)
    benchmark.setdefault('maps_4_512_3072_', 0) 
    benchmark['maps_4_512_3072_'] += t
    t = timeit.timeit(lambda: torch.sum(sumto_2048_768_, dim=-1), number=1)
    benchmark.setdefault('sumto_2048_768_', 0) 
    benchmark['sumto_2048_768_'] += t
    t = timeit.timeit(lambda: maps_1_512_768_1 + maps_1_512_768_2, number=1)
    benchmark.setdefault('maps_1_512_768_', 0) 
    benchmark['maps_1_512_768_'] += t
    t = timeit.timeit(lambda: torch.nn.functional.relu(maps_1_1_1_512_), number=1)
    benchmark.setdefault('maps_1_1_1_512_', 0) 
    benchmark['maps_1_1_1_512_'] += t
    t = timeit.timeit(lambda: maps_2_512_3072_1 + maps_2_512_3072_2, number=1)
    benchmark.setdefault('maps_2_512_3072_', 0) 
    benchmark['maps_2_512_3072_'] += t
    t = timeit.timeit(lambda: torch.sum(sumto_4_2_, dim=-1), number=1)
    benchmark.setdefault('sumto_4_2_', 0) 
    benchmark['sumto_4_2_'] += t
    t = timeit.timeit(lambda: torch.nn.functional.relu(maps_8_2_), number=1)
    benchmark.setdefault('maps_8_2_', 0) 
    benchmark['maps_8_2_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_8_12_512_512_1, matmul_8_12_512_512_2), number=1)
    benchmark.setdefault('matmul_8_12_512_512_', 0) 
    benchmark['matmul_8_12_512_512_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_4_12_512_64_1[0, :], torch.transpose(matmul_grad1_4_12_512_64_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_4_12_512_64_', 0) 
    benchmark['matmul_grad1_4_12_512_64_'] += t
    t = timeit.timeit(lambda: maps_4096_768_1 + maps_4096_768_2, number=1)
    benchmark.setdefault('maps_4096_768_', 0) 
    benchmark['maps_4096_768_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_2_2_1_1[0, :], torch.transpose(matmul_grad1_2_2_1_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_2_2_1_', 0) 
    benchmark['matmul_grad1_2_2_1_'] += t
    t = timeit.timeit(lambda: torch.nn.functional.relu(maps_4096_1_), number=1)
    benchmark.setdefault('maps_4096_1_', 0) 
    benchmark['maps_4096_1_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_16_12_512_64_1[0, :], torch.transpose(matmul_grad1_16_12_512_64_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_16_12_512_64_', 0) 
    benchmark['matmul_grad1_16_12_512_64_'] += t
    t = timeit.timeit(lambda: torch.nn.functional.relu(maps_8_1_1_512_), number=1)
    benchmark.setdefault('maps_8_1_1_512_', 0) 
    benchmark['maps_8_1_1_512_'] += t
    t = timeit.timeit(lambda: torch.matmul(torch.transpose(matmul_grad2_2_12_512_512_1, -1, -2), matmul_grad2_2_12_512_512_2[0, :]), number=1)
    benchmark.setdefault('matmul_grad2_2_12_512_512_', 0) 
    benchmark['matmul_grad2_2_12_512_512_'] += t
    t = timeit.timeit(lambda: torch.sum(sumto_2_2_, dim=-1), number=1)
    benchmark.setdefault('sumto_2_2_', 0) 
    benchmark['sumto_2_2_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_4_512_768_1_1[0, :], torch.transpose(matmul_grad1_4_512_768_1_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_4_512_768_1_', 0) 
    benchmark['matmul_grad1_4_512_768_1_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_2_512_3072_1_1[0, :], torch.transpose(matmul_grad1_2_512_3072_1_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_2_512_3072_1_', 0) 
    benchmark['matmul_grad1_2_512_3072_1_'] += t
    t = timeit.timeit(lambda: torch.sum(sumto_2_512_3072_, dim=-1), number=1)
    benchmark.setdefault('sumto_2_512_3072_', 0) 
    benchmark['sumto_2_512_3072_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_2_512_768_1_1[0, :], torch.transpose(matmul_grad1_2_512_768_1_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_2_512_768_1_', 0) 
    benchmark['matmul_grad1_2_512_768_1_'] += t
    t = timeit.timeit(lambda: torch.matmul(torch.transpose(matmul_grad2_4_12_512_512_1, -1, -2), matmul_grad2_4_12_512_512_2[0, :]), number=1)
    benchmark.setdefault('matmul_grad2_4_12_512_512_', 0) 
    benchmark['matmul_grad2_4_12_512_512_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_16_512_3072_1_1[0, :], torch.transpose(matmul_grad1_16_512_3072_1_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_16_512_3072_1_', 0) 
    benchmark['matmul_grad1_16_512_3072_1_'] += t
    t = timeit.timeit(lambda: maps_2048_768_1 + maps_2048_768_2, number=1)
    benchmark.setdefault('maps_2048_768_', 0) 
    benchmark['maps_2048_768_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_1_512_3072_1_1[0, :], torch.transpose(matmul_grad1_1_512_3072_1_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_1_512_3072_1_', 0) 
    benchmark['matmul_grad1_1_512_3072_1_'] += t
    t = timeit.timeit(lambda: torch.sum(sum_4096_768_, dim=1), number=1)
    benchmark.setdefault('sum_4096_768_', 0) 
    benchmark['sum_4096_768_'] += t
    t = timeit.timeit(lambda: torch.sum(sum_1024_768_, dim=1), number=1)
    benchmark.setdefault('sum_1024_768_', 0) 
    benchmark['sum_1024_768_'] += t
    t = timeit.timeit(lambda: maps_1024_768_1 + maps_1024_768_2, number=1)
    benchmark.setdefault('maps_1024_768_', 0) 
    benchmark['maps_1024_768_'] += t
    t = timeit.timeit(lambda: torch.sum(sumto_16_512_768_, dim=-1), number=1)
    benchmark.setdefault('sumto_16_512_768_', 0) 
    benchmark['sumto_16_512_768_'] += t
    t = timeit.timeit(lambda: torch.matmul(torch.transpose(matmul_grad2_8_12_512_64_1, -1, -2), matmul_grad2_8_12_512_64_2[0, :]), number=1)
    benchmark.setdefault('matmul_grad2_8_12_512_64_', 0) 
    benchmark['matmul_grad2_8_12_512_64_'] += t
    t = timeit.timeit(lambda: torch.sum(sum_4_12_512_512_, dim=3), number=1)
    benchmark.setdefault('sum_4_12_512_512_', 0) 
    benchmark['sum_4_12_512_512_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_1_12_512_64_1, matmul_1_12_512_64_2), number=1)
    benchmark.setdefault('matmul_1_12_512_64_', 0) 
    benchmark['matmul_1_12_512_64_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_1_12_512_512_1[0, :], torch.transpose(matmul_grad1_1_12_512_512_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_1_12_512_512_', 0) 
    benchmark['matmul_grad1_1_12_512_512_'] += t
    t = timeit.timeit(lambda: torch.matmul(torch.transpose(matmul_grad2_768_3072_1, -1, -2), matmul_grad2_768_3072_2[0, :]), number=1)
    benchmark.setdefault('matmul_grad2_768_3072_', 0) 
    benchmark['matmul_grad2_768_3072_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_16_12_512_512_1, matmul_16_12_512_512_2), number=1)
    benchmark.setdefault('matmul_16_12_512_512_', 0) 
    benchmark['matmul_16_12_512_512_'] += t
    t = timeit.timeit(lambda: maps_16_512_3072_1 + maps_16_512_3072_2, number=1)
    benchmark.setdefault('maps_16_512_3072_', 0) 
    benchmark['maps_16_512_3072_'] += t
    t = timeit.timeit(lambda: torch.nn.functional.relu(maps_16_2_), number=1)
    benchmark.setdefault('maps_16_2_', 0) 
    benchmark['maps_16_2_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_8_2_1_1[0, :], torch.transpose(matmul_grad1_8_2_1_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_8_2_1_', 0) 
    benchmark['matmul_grad1_8_2_1_'] += t
    t = timeit.timeit(lambda: torch.matmul(torch.transpose(matmul_grad2_16_12_512_512_1, -1, -2), matmul_grad2_16_12_512_512_2[0, :]), number=1)
    benchmark.setdefault('matmul_grad2_16_12_512_512_', 0) 
    benchmark['matmul_grad2_16_12_512_512_'] += t
    t = timeit.timeit(lambda: torch.sum(sumto_2_512_768_, dim=-1), number=1)
    benchmark.setdefault('sumto_2_512_768_', 0) 
    benchmark['sumto_2_512_768_'] += t
    t = timeit.timeit(lambda: torch.nn.functional.relu(maps_4_1_1_512_), number=1)
    benchmark.setdefault('maps_4_1_1_512_', 0) 
    benchmark['maps_4_1_1_512_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_8_512_768_1_1[0, :], torch.transpose(matmul_grad1_8_512_768_1_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_8_512_768_1_', 0) 
    benchmark['matmul_grad1_8_512_768_1_'] += t
    t = timeit.timeit(lambda: maps_30522_768_1 + maps_30522_768_2, number=1)
    benchmark.setdefault('maps_30522_768_', 0) 
    benchmark['maps_30522_768_'] += t
    t = timeit.timeit(lambda: torch.matmul(torch.transpose(matmul_grad2_1_12_512_512_1, -1, -2), matmul_grad2_1_12_512_512_2[0, :]), number=1)
    benchmark.setdefault('matmul_grad2_1_12_512_512_', 0) 
    benchmark['matmul_grad2_1_12_512_512_'] += t
    t = timeit.timeit(lambda: torch.sum(sumto_1_512_768_, dim=-1), number=1)
    benchmark.setdefault('sumto_1_512_768_', 0) 
    benchmark['sumto_1_512_768_'] += t
    t = timeit.timeit(lambda: maps_2_512_768_1 + maps_2_512_768_2, number=1)
    benchmark.setdefault('maps_2_512_768_', 0) 
    benchmark['maps_2_512_768_'] += t
    t = timeit.timeit(lambda: torch.sum(sumto_4_512_3072_, dim=-1), number=1)
    benchmark.setdefault('sumto_4_512_3072_', 0) 
    benchmark['sumto_4_512_3072_'] += t
    t = timeit.timeit(lambda: torch.matmul(torch.transpose(matmul_grad2_16_12_512_64_1, -1, -2), matmul_grad2_16_12_512_64_2[0, :]), number=1)
    benchmark.setdefault('matmul_grad2_16_12_512_64_', 0) 
    benchmark['matmul_grad2_16_12_512_64_'] += t
    t = timeit.timeit(lambda: maps_8192_768_1 + maps_8192_768_2, number=1)
    benchmark.setdefault('maps_8192_768_', 0) 
    benchmark['maps_8192_768_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_8_12_512_64_1, matmul_8_12_512_64_2), number=1)
    benchmark.setdefault('matmul_8_12_512_64_', 0) 
    benchmark['matmul_8_12_512_64_'] += t
    t = timeit.timeit(lambda: torch.matmul(matmul_grad1_16_2_1_1[0, :], torch.transpose(matmul_grad1_16_2_1_2[0, :], -1, -2)), number=1)
    benchmark.setdefault('matmul_grad1_16_2_1_', 0) 
    benchmark['matmul_grad1_16_2_1_'] += t
    t = timeit.timeit(lambda: torch.sum(sumto_1_512_3072_, dim=-1), number=1)
    benchmark.setdefault('sumto_1_512_3072_', 0) 
    benchmark['sumto_1_512_3072_'] += t
    t = timeit.timeit(lambda: torch.sum(sumto_4096_768_, dim=-1), number=1)
    benchmark.setdefault('sumto_4096_768_', 0) 
    benchmark['sumto_4096_768_'] += t
rust_code = '['
for uid in benchmark:
    rust_code += '("' + uid + '", ' + str(benchmark[uid] / 2) + '), '
rust_code += ']'
print(rust_code)
