mlp_bgq:
	mpixlc main.c -o mlp.xl -lm
mlp_kratos:
	mpicc -Wall main.c -o mlp.out -lm
exp_bgq:
	mpixlc experiments.c -o exp.xl -lm
exp_bgq_io:
	mpixlc -DALL_SAMPLES experiments.c -o exp_io.xl -lm
exp_kratos:
	mpicc -Wall -DKRATOS experiments.c -o exp.out -lm
exp_kratos_async:
	mpicc -Wall -DKRATOS -DASYNC experiments.c -o exp_async.out -lm
