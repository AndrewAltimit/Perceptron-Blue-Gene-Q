#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

// -D ALL_SAMPLES to have each rank read all samples
// -D KRATOS to run on kratos
// -D ASYNC to use non-blocking communications (only on kratos)

#ifdef KRATOS

#define get_time() MPI_Wtime()
#define MPI_TIME_T MPI_DOUBLE
typedef double tick_t;

#else

#include <hwi/include/bqc/A2_inlines.h>
#define get_time() GetTimeBase()
#define MPI_TIME_T MPI_UNSIGNED_LONG_LONG
typedef unsigned long long tick_t;

#endif

/************************************* Global Variables *************************************/
int nlayers = 2; /* number of layers */
int ninputs; /* number of inputs to the network (basically layer 0) */
int *nnodes; /* number of neurons in each layer */
int *nproc; /* number of ranks needed to process each layer */

double eta = 0.01; /* learning rate */

double *dEdz; /* changes in output error w.r.t. neuron outputs */
double *input_activations;
double **activations; /* neuron activations: z = sigmoid(neuron bias) */
double **errors; /* partial error signals, sum to dEdz */
double ***weights;
double **biases;

int rank;
int nranks;
int nodes_per_rank = 1;
int offset; /* node offset for this rank */

#ifdef ASYNC
MPI_Request *requests;
#endif

/*************************************** Inner Product **************************************/
double inner_product(int n, double *a_vec, double *b_vec) {
    int i;
    double output = 0;
    for (i = 0; i < n; i++) {
        output += a_vec[i]*b_vec[i];
    }
    return output;
}

/*************************************** Read Samples ***************************************/
/* Input: filename
   Output: samples - the data for my rank
           nsamples - total number of samples in the datafile
           samples_per_rank - max number of samples stored per rank */
unsigned char *read_samples(char *filename, int *ns_out, int *spr_out) {
    MPI_File fh;
    MPI_Offset file_size;
    
    /* Opeen the file */
    int rc = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (rc != MPI_SUCCESS) {
        fprintf(stderr, "ERROR: unable to open file '%s'\n", filename);
        exit(EXIT_FAILURE);
    }
    MPI_File_get_size(fh, &file_size);

    /* Determine how to divide up the file and samples */
    int sample_size = ninputs + nnodes[nlayers-1]; /* sample stride (features + outputs) */
    if ((file_size % sample_size) != 0) {
        fprintf(stderr, "ERROR: sizes inconsistent between network and file '%s'\n", filename);
        exit(EXIT_FAILURE);
    }
    int nsamples = file_size/sample_size; /* number of data samples */
#ifdef ALL_SAMPLES
    int samples_per_rank = nsamples;
    unsigned char *samples = malloc(nsamples*sample_size*sizeof(unsigned char));
    MPI_File_read_at_all(fh, 0, samples, nsamples*sample_size, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
#else
    int samples_per_rank = 1 + ((nsamples - 1) / nranks); /* number of samples per rank (rounded up) */

    /* Determine range of samples to read (may be 0) */
    int my_first_sample = samples_per_rank*rank;
    if (my_first_sample > nsamples) {
        my_first_sample = nsamples;
    }
    int my_last_sample = samples_per_rank*(rank + 1) - 1;
    if (my_last_sample > nsamples) {
        my_last_sample = nsamples;
    }
    int my_num_samples = my_last_sample - my_first_sample + 1;

    /* Read samples */
    unsigned char *samples = malloc(my_num_samples*sample_size*sizeof(unsigned char));
    MPI_File_read_at_all(fh, my_first_sample*sample_size, samples, my_num_samples*sample_size, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
#endif
    MPI_File_close(&fh);


    *ns_out = nsamples;
    *spr_out = samples_per_rank;
    return samples;
}


/* INPUT: <num neurons per layer> */
int main(int argc, char *argv[])
{
    int i,j,k,s,r; /* loop counters */

    /* MPI startup */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

/************************************ Network Dimensions ************************************/
    /* Parse layer sizes from command-line input */
    if (argc != 2) {
        fprintf(stderr, "ERROR: incorrect number if command-line inputs\n");
        return EXIT_FAILURE;
    }
    nnodes = calloc(nlayers, sizeof(int));
    nproc = calloc(nlayers, sizeof(int));

    /* Number neurons per layer */
    ninputs = atoi(argv[1]);
    if ((ninputs == 0) || ((ninputs % nranks) != 0)) {
        fprintf(stderr, "ERROR: invalid layer size '%s'\n", argv[3]);
        return EXIT_FAILURE;
    }
    for (i = 0; i < nlayers; i++) {
        nnodes[i] = ninputs;
        nproc[i] = nranks;
    }
    nodes_per_rank = ninputs/nranks;
    offset = nodes_per_rank*rank;

    int niter = 2000;
    char filename[100];
#ifdef KRATOS
    sprintf(filename,"/home/parallel/2017/PPChorakp/data/%d_train_bytes", ninputs);
#else
    sprintf(filename,"/gpfs/u/scratch/PCP6/PCP6hrkp/data/%d_train_bytes", ninputs);
#endif

/************************************** Allocate Network ************************************/
    /* Allocate assorted arrays */
#ifdef ASYNC
    requests = calloc(nranks, sizeof(MPI_Request));
#endif
    dEdz = calloc(nodes_per_rank, sizeof(double));
    input_activations = calloc(ninputs, sizeof(double));

    /* Allocate - activations for all nodes each layer
                - error derivative signals for all nodes in each layer
                - input weights for each node for which this rank is responsible
                - the bias for each node for which this rank is responsible */
    activations = calloc(nlayers, sizeof(double *));
    errors = calloc(nlayers, sizeof(double *));
    weights = calloc(nlayers, sizeof(double **));
    biases = calloc(nlayers, sizeof(double *));
    int nprev;
    /* Loop over layers */
    for (i = 0; i < nlayers; i++) {
        activations[i] = calloc(nnodes[i], sizeof(double));
        errors[i] = calloc(nnodes[i], sizeof(double));
        if (rank < nproc[i]) {
            /* My rank is within those needed for layer i */
            weights[i] = calloc(nodes_per_rank, sizeof(double *));
            biases[i] = calloc(nodes_per_rank, sizeof(double));
            nprev = (i > 0) ? nnodes[i-1] : ninputs; /* size of the previous layer */
            /* Nodes for which I am responsible */
            for (j = 0; j < nodes_per_rank; j++) {
                srand(offset+j+i*nnodes[i]); /* seed specific to each node */
                /* Initialize its bias */
                biases[i][j] = 2.0*(((double)rand())/RAND_MAX - 0.5); /* TODO: randomize? */
                /* Allocate and initialize its input weights */
                weights[i][j] = calloc(nprev, sizeof(double));
                for (k = 0; k < nprev; k++) {
                    weights[i][j][k] = 2.0*(((double)rand())/RAND_MAX - 0.5); /* TODO: randomize? */
                }
            }
        } else {
            /* Rank not active for layer i */
            weights[i] = NULL;
            biases[i] = NULL;
        }
    }

    /* Variables related to training and testing data */
    int sample; /* sample counter */
    int sample_size = ninputs + nnodes[nlayers-1]; /* sample data size or length */
    int nsamples; /* total number of samples */
    int samples_per_rank; /* max number of samples stored per rank */
    unsigned char *samples; /* all samples stored by my rank */
    unsigned char *sample_data; /* current sample data */
    sample_data = calloc(sample_size, sizeof(unsigned char));

    /* Timing variables */
    tick_t init_time, start_time, end_time;
    tick_t t1_start, t2_start; /* timers */

    tick_t *share_times = calloc(niter, sizeof(tick_t));
    tick_t *forward_times = calloc(niter, sizeof(tick_t));
    tick_t *bcast_times = calloc(niter, sizeof(tick_t));
    tick_t *backward_times = calloc(niter, sizeof(tick_t));
    tick_t *reduce_times = calloc(niter, sizeof(tick_t));

    tick_t *share_comb = calloc(niter, sizeof(tick_t));
    tick_t *forward_comb = calloc(niter, sizeof(tick_t));
    tick_t *bcast_comb = calloc(niter, sizeof(tick_t));
    tick_t *backward_comb = calloc(niter, sizeof(tick_t));
    tick_t *reduce_comb = calloc(niter, sizeof(tick_t));

/*************************************** Train Network **************************************/
    double node_bias;
    double node_out;
    double dEdb; /* derivatives of output error w.r.t. node biases */

    // Start time (finished initializing)
    MPI_Barrier(MPI_COMM_WORLD);
    init_time = get_time();

    samples = read_samples(filename, &nsamples, &samples_per_rank);

    // Start time (finished reading training data)
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = get_time();

    // Loop over all training data
    for (s = 0; s < niter; s++) {
        sample = (s % nsamples);

        t1_start = get_time();
#ifdef ALL_SAMPLES
        memcpy(sample_data, &samples[sample*sample_size], sample_size);
#else
        // sample / samples_per_rank -> the rank which has the sample data
        if (rank == (sample / samples_per_rank)) {
            // sample % samples_per_rank -> index at which the rank stored it
            memcpy(sample_data, &samples[(sample % samples_per_rank)*sample_size], sample_size);
        }
        // Send the current sample to all ranks
        MPI_Bcast(sample_data, sample_size, MPI_UNSIGNED_CHAR, sample/samples_per_rank, MPI_COMM_WORLD);
#endif
        share_times[s] = get_time() - t1_start;

        /* * * FORWARD PROPAGATION * * */
        // Convert inputs to doubles
        for (j = 0; j < ninputs; j++) {
            input_activations[j] = (double)sample_data[j];
        }

        // Set activations I'm responsible for based on input 
        if (rank < nproc[0]) {
            for (j = 0; j < nodes_per_rank; j++) {
                node_bias = biases[0][j] + inner_product(ninputs, input_activations, weights[0][j]);
                activations[0][offset+j] = 1.0/(1.0 + exp(-node_bias));
            }
        }

        // Loop over layers
        t1_start = get_time();
        for (i = 1; i < nlayers; i++) {
            // Send and receive activations
            t2_start = get_time();
            for (r = 0; r < nproc[i-1]; r++) {
#ifdef ASYNC
                MPI_Ibcast(&activations[i-1][r*nodes_per_rank], nodes_per_rank, MPI_DOUBLE, r, MPI_COMM_WORLD, &requests[r]);
#else
                MPI_Bcast(&activations[i-1][r*nodes_per_rank], nodes_per_rank, MPI_DOUBLE, r, MPI_COMM_WORLD);
#endif
            }
#ifdef ASYNC
            MPI_Waitall(nproc[i-1], requests, MPI_STATUSES_IGNORE);
#endif
            bcast_times[s] = get_time() - t2_start;
            // Update my nodes
            if (rank < nproc[i]) {
                for (j = 0; j < nodes_per_rank; j++) {
                    node_bias = biases[i][j] + inner_product(nnodes[i-1], activations[i-1], weights[i][j]);
                    activations[i][offset+j] = 1.0/(1.0 + exp(-node_bias));
                }
            }
        }
        forward_times[s] = get_time() - t1_start;

        /* * * BACKWARD PROPAGATION * * */
        // Derivatives of the output error with respect to network outputs
        if (rank < nproc[nlayers-1]) {
            for (j = 0; j < nodes_per_rank; j++) {
                dEdz[j] = activations[nlayers-1][offset+j] - (double)sample_data[ninputs+offset+j]; // dEdz
            }
        }

        // Loop over layers
        t1_start = get_time();
        for (i = nlayers-1; i > 0; i--) {
            // Zero for all ranks because reduction (later) includes all
            bzero(errors[i-1], nnodes[i-1]*sizeof(double));
            // Update my nodes
            if (rank < nproc[i]) {
                for (j = 0; j < nodes_per_rank; j++) {
                    node_out = activations[i][offset+j]; // z
                    dEdb = dEdz[j]*(node_out*(1-node_out)); // dEdb = dEdz*dzdb = dEdz*z*(1-z)
                    for (k = 0; k < nnodes[i-1]; k++) {
                        errors[i-1][k] += dEdb*weights[i][j][k]; /* propogate error signals backward */
                        weights[i][j][k] -= eta*dEdb*activations[i-1][k];
                    }
                    biases[i][j] -= eta*dEdb;
                }
            }
            // Sum up error derivatives across ranks
            t2_start = get_time();
            for (r = 0; r < nproc[i-1]; r++) {
#ifdef ASYNC
                MPI_Ireduce(&errors[i-1][r*nodes_per_rank], dEdz, nodes_per_rank, MPI_DOUBLE, MPI_SUM, r, MPI_COMM_WORLD, &requests[r]);
#else
                MPI_Reduce(&errors[i-1][r*nodes_per_rank], dEdz, nodes_per_rank, MPI_DOUBLE, MPI_SUM, r, MPI_COMM_WORLD);
#endif
            }
#ifdef ASYNC
            MPI_Waitall(nproc[i-1], requests, MPI_STATUSES_IGNORE);
#endif
            reduce_times[s] = get_time() - t2_start;
        }
        backward_times[s] = get_time() - t1_start;

        // Special case for the first layer
        if (rank < nproc[0]) {
            for (j = 0; j < nodes_per_rank; j++) {
                node_out = activations[0][offset+j]; // z
                dEdb = dEdz[j]*(node_out*(1-node_out)); // dEdb = dEdz*dzdb = dEdz*z*(1-z)
                for (k = 0; k < ninputs; k++) {
                    weights[0][j][k] -= eta*dEdb*input_activations[k];
                }
                biases[0][j] -= eta*dEdb;
            }
        }
    }
    free(samples);

    // End time (finished training network)
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = get_time();

/**************************************** Print Results *************************************/
    tick_t max_share_time = 0; tick_t min_share_time = 0;
    tick_t max_forward_time = 0; tick_t min_forward_time = 0;
    tick_t max_bcast_time = 0; tick_t min_bcast_time = 0;
    tick_t max_backward_time = 0; tick_t min_backward_time = 0;
    tick_t max_reduce_time = 0; tick_t min_reduce_time = 0;

    MPI_Reduce(share_times, share_comb, niter, MPI_TIME_T, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(forward_times, forward_comb, niter, MPI_TIME_T, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(bcast_times, bcast_comb, niter, MPI_TIME_T, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(backward_times, backward_comb, niter, MPI_TIME_T, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(reduce_times, reduce_comb, niter, MPI_TIME_T, MPI_MAX, 0, MPI_COMM_WORLD);

    for (s = 0; s < niter; s++) {
        max_share_time += share_comb[s];
        max_forward_time += forward_comb[s];
        max_bcast_time += bcast_comb[s];
        max_backward_time += backward_comb[s];
        max_reduce_time += reduce_comb[s];
    }

    MPI_Reduce(share_times, share_comb, niter, MPI_TIME_T, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(forward_times, forward_comb, niter, MPI_TIME_T, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(bcast_times, bcast_comb, niter, MPI_TIME_T, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(backward_times, backward_comb, niter, MPI_TIME_T, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(reduce_times, reduce_comb, niter, MPI_TIME_T, MPI_MIN, 0, MPI_COMM_WORLD);

    for (s = 0; s < niter; s++) {
        min_share_time += share_comb[s];
        min_forward_time += forward_comb[s];
        min_bcast_time += bcast_comb[s];
        min_backward_time += backward_comb[s];
        min_reduce_time += reduce_comb[s];
    }

    //Print experiment information, time, and network performance
    if (rank == 0) {
#ifdef KRATOS
        printf("%d, %d, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", \
            nranks, ninputs, start_time - init_time, end_time - start_time, \
            max_share_time, max_forward_time, max_bcast_time, max_backward_time, max_reduce_time, \
            min_share_time, min_forward_time, min_bcast_time, min_backward_time, min_reduce_time);
#else
        printf("%d, %d, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", \
            nranks, ninputs, (start_time - init_time)/1.6E9, (end_time - start_time)/1.6E9, \
            max_share_time/1.6E9, max_forward_time/1.6E9, max_bcast_time/1.6E9, max_backward_time/1.6E9, max_reduce_time/1.6E9, \
            min_share_time/1.6E9, min_forward_time/1.6E9, min_bcast_time/1.6E9, min_backward_time/1.6E9, min_reduce_time/1.6E9);
#endif
    }

/**************************************** Free Memory ***************************************/
    free(reduce_times); free(reduce_comb);
    free(backward_times); free(backward_comb);
    free(bcast_times); free(bcast_comb);
    free(forward_times); free(forward_comb);
    free(share_times); free(share_comb);

    free(sample_data);
    
    for (i = 0; i < nlayers; i++) {
        if (rank < nproc[i]) {
            for (j = 0; j < nodes_per_rank; j++) {
                free(weights[i][j]);
            }
            free(biases[i]);
            free(weights[i]);
        }
        free(errors[i]);
        free(activations[i]);
    }
    free(biases);
    free(weights);
    free(errors);
    free(activations);

    free(input_activations);
    free(dEdz);
#ifdef ASYNC
    free(requests);
#endif
    free(nproc);
    free(nnodes);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

