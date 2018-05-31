#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

/************************************* Global Variables *************************************/
int nlayers; /* number of layers */
int ninputs; /* number of inputs to the network (basically layer 0) */
int *nnodes; /* number of neurons in each layer */
int *nproc; /* number of ranks needed to process each layer */

double eta; /* learning rate */

double *dEdz; /* changes in output error w.r.t. neuron outputs */
double *input_activations;
double **activations; /* neuron activations: z = sigmoid(neuron bias) */
double **errors; /* partial error signals, sum to dEdz */
double ***weights;
double **biases;

int rank;
int nranks;
int nodes_per_rank = 1; /* neurons per rank */
int offset; /* neuron offset for this rank */

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
    MPI_File_close(&fh);

    *ns_out = nsamples;
    *spr_out = samples_per_rank;
    return samples;
}

/************************************ Forward Propogation ***********************************/
void forward_prop(unsigned char *sample_data) {
    int i,j,r; /* loop counters */
    double node_bias;

    // Convert inputs to doubles
    for (j = 0; j < ninputs; j++) {
        input_activations[j] = (double)sample_data[j]/255.0; /* scale the input to range [0,1] */
    }

    // Set activations I'm responsible for based on input 
    if (rank < nproc[0]) {
        for (j = 0; j < nodes_per_rank; j++) {
            node_bias = biases[0][j] + inner_product(ninputs, input_activations, weights[0][j]);
            activations[0][offset+j] = 1.0/(1.0 + exp(-node_bias));
        }
    }

    // Loop over layers 
    for (i = 1; i < nlayers; i++) {
        // Send and receive activations
        for (r = 0; r < nproc[i-1]; r++) {
            MPI_Bcast(&activations[i-1][r*nodes_per_rank], nodes_per_rank, MPI_DOUBLE, r, MPI_COMM_WORLD);
        }
        // Update my nodes
        if (rank < nproc[i]) {
            for (j = 0; j < nodes_per_rank; j++) {
                node_bias = biases[i][j] + inner_product(nnodes[i-1], activations[i-1], weights[i][j]);
                activations[i][offset+j] = 1.0/(1.0 + exp(-node_bias));
            }
        }
    }
}

/************************************ Backward Propogation **********************************/
void backward_prop(unsigned char *sample_data) {
    int i,j,k,r; /* loop counters */
    double node_out;
    double dEdb; /* derivatives of output error w.r.t. node biases */

    // Derivatives of the output error with respect to network outputs
    if (rank < nproc[nlayers-1]) {
        for (j = 0; j < nodes_per_rank; j++) {
            dEdz[j] = activations[nlayers-1][offset+j] - (double)sample_data[ninputs+offset+j]; // dEdz
        }
    }

    // Loop over layers
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
        for (r = 0; r < nproc[i-1]; r++) {
            MPI_Reduce(&errors[i-1][r*nodes_per_rank], dEdz, nodes_per_rank, MPI_DOUBLE, MPI_SUM, r, MPI_COMM_WORLD);
        }
    }

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


/* USAGE: mpirun -np <num ranks> ./a.out <training data filename> <test data filename> <learning rate> <numtraining iterations> <num inputs> <num nodes in layer 1> ... <num nodes in layer n> */
int main(int argc, char *argv[])
{
    int i,j,k,r; /* loop counters */

    double init_time;
    double start_time;
    double end_time;

    /* MPI startup */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

/************************************ Network Dimensions ************************************/
    /* Parse layer sizes from command-line input */
    nlayers = argc - 6;
    if (nlayers < 1) {
        fprintf(stderr, "ERROR: must specify at least an input and output layer\n");
        return EXIT_FAILURE;
    }
    nnodes = calloc(nlayers, sizeof(int));
    nproc = calloc(nlayers, sizeof(int));
    /* Number of inputs */
    ninputs = atoi(argv[5]);
    if (ninputs == 0) {
        fprintf(stderr, "ERROR: input layer invalid size '%s'\n", argv[5]);
        return EXIT_FAILURE;
    }
    /* Number of nodes per layer */
    int nmax = 0; // maximum number of nodes in a layer
    for (i = 0; i < nlayers; i++) {
        nnodes[i] = atoi(argv[i+6]);
        if (nnodes[i] == 0) {
            fprintf(stderr, "ERROR: layer %d invalid size '%s'\n", i+1, argv[i+6]);
            return EXIT_FAILURE;
        } else if (nnodes[i] > nmax) {
            nmax = nnodes[i];
        }
    }

    /* Check that the layer sizes work with the number of ranks */
    if ((nmax % nranks) != 0) {
        fprintf(stderr, "ERROR: cannot evenly divide %d nodes over %d ranks\n", nmax, nranks);
        return EXIT_FAILURE;
    }
    nodes_per_rank = nmax/nranks;
    offset = nodes_per_rank*rank;
    for (i = 0; i < nlayers; i++) {
        if ((nnodes[i] % nodes_per_rank) != 0) {
            fprintf(stderr, "ERROR: cannot evenly divide %d nodes into groups of %d\n", nnodes[i], nodes_per_rank);
            return EXIT_FAILURE;
        }
        nproc[i] = nnodes[i]/nodes_per_rank;
    }

    /* Learning rate */
    eta = atof(argv[3]);
    if (eta == 0.0) {
        fprintf(stderr, "ERROR: invalid learning rate '%s'\n", argv[3]);
        return EXIT_FAILURE;
    }

    /* Number of training iterations */
    int niter = atoi(argv[4]);
    if (niter == 0) {
        fprintf(stderr, "ERROR: invalid number of training iterations '%s'\n", argv[4]);
        return EXIT_FAILURE;
    }

/************************************** Allocate Network ************************************/
    /* Allocate assorted arrays */
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

/*************************************** Train Network **************************************/
    // Start time (finished initializing)
    MPI_Barrier(MPI_COMM_WORLD);
    init_time = MPI_Wtime();

    samples = read_samples(argv[1], &nsamples, &samples_per_rank);

    // Start time (finished reading training data)
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    double err, err_sum;
    double *training_error;
    training_error = calloc(niter, sizeof(double));

    // Loop over all training data
    srand(1); /* synchronize random number generator across ranks */
    for (k = 0; k < niter; k++) {
//      sample = (rand() % nsamples);
        sample = (k % nsamples);
        // sample / samples_per_rank -> the rank which has the sample data
        if (rank == (sample / samples_per_rank)) {
            // sample % samples_per_rank -> index at which the rank stored it
            memcpy(sample_data, &samples[(sample % samples_per_rank)*sample_size], sample_size);
        }
        // Send the current sample to all ranks
        MPI_Bcast(sample_data, sample_size, MPI_UNSIGNED_CHAR, sample/samples_per_rank, MPI_COMM_WORLD);

        // Run the network forward and backward
        forward_prop(sample_data);
        backward_prop(sample_data);

        // Calculate output error
        err_sum = 0.0;
        if (rank < nproc[nlayers-1]) {
            for (j = 0; j < nodes_per_rank; j++) {
                err = activations[nlayers-1][offset+j] - (double)sample_data[ninputs+offset+j];
                err_sum += 0.5*err*err;
            }
        }
        // Sum errors across nodes
        MPI_Reduce(&err_sum, &training_error[k], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    free(samples);

    // End time (finished training network)
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

/**************************************** Test Network **************************************/
    samples = read_samples(argv[2], &nsamples, &samples_per_rank);

    int ncorrect = 0;
    int max_loc, label;

    // Loop over all testing data
    for (sample = 0; sample < nsamples; sample++) {
        // sample / samples_per_rank -> the rank which has the sample data
        if (rank == (sample / samples_per_rank)) {
            // sample % samples_per_rank -> index at which the rank stored it
            memcpy(sample_data, &samples[(sample % samples_per_rank)*sample_size], sample_size);
        }
        // Send the current sample to all ranks
        MPI_Bcast(sample_data, sample_size, MPI_UNSIGNED_CHAR, sample/samples_per_rank, MPI_COMM_WORLD);

        // Run the network forward
        forward_prop(sample_data);

        /* Calculate output accuracy */
        // Sync output layer across ranks
        for (r = 0; r < nproc[nlayers-1]; r++) {
            MPI_Bcast(&activations[nlayers-1][r*nodes_per_rank], nodes_per_rank, MPI_DOUBLE, r, MPI_COMM_WORLD);
        }

        // Find which node has the greatest activation
        max_loc = 0;
        for (j = 0; j < nnodes[nlayers-1]; j++) {
            if (activations[nlayers-1][j] > activations[nlayers-1][max_loc]) {
                max_loc = j;
            }
            if (sample_data[ninputs+j] > 0) {
                label = j;
            }
        }

        // Does it match the one-hot label?
        if (max_loc == label) {
            ncorrect++;
        }
    }

    free(samples);

/**************************************** Print Results *************************************/
    //Print experiment information, time, and network performance
    if (rank == 0) {
        printf("Ranks: %d\n", nranks);
        printf("Network: %d, ", ninputs);
        for (i = 0; i < nlayers; i++) {
            printf("%d, ", nnodes[i]);
        }
        printf("\n");
        printf("File I/O Time: %lf\n", start_time - init_time);
        printf("Training Time: %lf\n", end_time - start_time);
        printf("Accuracy: %lf\n", ((double)ncorrect)/nsamples);

        FILE *fp;
        if ((fp = fopen("training_error.txt","w")) == NULL) {
            perror("");
        } else {
            for (k = 0; k < niter; k++) {
                fprintf(fp, "%lf\n", training_error[k]);
            }
        }
    }
    free(training_error);

/**************************************** Free Memory ***************************************/
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

    free(nproc);
    free(nnodes);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

