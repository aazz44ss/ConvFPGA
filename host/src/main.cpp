#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <fstream>
#include <string>
#include <chrono>

#include <CL/cl.h>
#include "AOCLUtils/aocl_utils.h"

#include "hdf5.h"

#define KERNEL_PARALLEL 4
#define FILTER_PARALLEL 4

#define Winograd_Index 3
#define Winograd_output 2
#define use_relu 1

bool remote_duration = 1;
int mini_batch = 1;

int platform = 3;
int num_devices;

int frac_ptr = 0;
int fraction[][3] = {
	{ 7,8 },//L0
	{ 7,8 },//L1
	{ 8,8 },//L2
	{ 7,8 },//L3
	{ 5,8 },//L4
	{ 5,8 },//L5
	{ 5,7 }, //L6
	{ 5,7 }, //L7
	{ 0,7 }, //L8
};

#define Fixed_point 1
#if Fixed_point
typedef char DPTYPE;
typedef short BUFTYPE;
typedef int MACTYPE;
#define MASK 0xFF       // used for MAC, FP8
#define MASK_SIGN 0x80  // used for relu, FP8
#define DPTYPE_HI 0x7F
#define DPTYPE_LO 0x80
#define MASK_MAC 0x01FFFFFF
#else
typedef float DPTYPE;
typedef float BUFTYPE;
typedef float MACTYPE;
#define DPTYPE_HI INFINITY
#define DPTYPE_LO -INFINITY
#define MASK_MAC 0xFFFFFFFF
#endif

#if KERNEL_PARALLEL==FILTER_PARALLEL
#define SAME_PARALLEL 1
#define KERNEL_MAIN 0
#define PARALLEL KERNEL_PARALLEL
#elif KERNEL_PARALLEL>FILTER_PARALLEL
#define KERNEL_MAIN 1
#define on_RD 1
#if on_RD
#define PARALLEL KERNEL_PARALLEL
#else
#define PARALLEL FILTER_PARALLEL
#endif
#else
#define FILTER_MAIN 1
#endif

void cleanup();
static cl_context context;
static cl_platform_id *platforms = NULL;
static cl_device_id *devices = NULL;
static cl_program *program = NULL;
static cl_command_queue *command_queue = NULL;
static cl_command_queue *weightRD_queue = NULL;
static cl_command_queue *biasRD_queue = NULL;
static cl_command_queue *dataRD_queue = NULL;
static cl_command_queue *dataWR_queue = NULL;

static cl_event event[16][9];

static cl_kernel Conv_kernel;
static cl_kernel weightRD_kernel;
static cl_kernel biasRD_kernel;
static cl_kernel dataRD_kernel;
static cl_kernel dataWR_kernel;

cl_mem *Z0_mem = NULL;
cl_mem *W1_conv_mem = NULL, *b1_conv_mem = NULL, *Z1_pool_mem = NULL;
cl_mem *W2_conv_mem = NULL, *b2_conv_mem = NULL, *Z2_pool_mem = NULL;
cl_mem *W3_mem = NULL, *b3_mem = NULL, *Z3_mem = NULL;
cl_mem *W4_mem = NULL, *b4_mem = NULL, *Z4_mem = NULL;
cl_mem *W5_mem = NULL, *b5_mem = NULL, *Z5_mem = NULL;

cl_mem *W3_conv_mem = NULL, *b3_conv_mem = NULL, *Z3_conv_mem = NULL;
cl_mem *W4_conv_mem = NULL, *b4_conv_mem = NULL, *Z4_conv_mem = NULL;
cl_mem *W5_conv_mem = NULL, *b5_conv_mem = NULL, *Z5_pool_mem = NULL;
cl_mem *W6_mem = NULL, *b6_mem = NULL, *Z6_mem = NULL;
cl_mem *W7_mem = NULL, *b7_mem = NULL, *Z7_mem = NULL;
cl_mem *W8_mem = NULL, *b8_mem = NULL, *Z8_mem = NULL;

std::chrono::system_clock::time_point tic, toc;
std::chrono::system_clock::time_point tic2, toc2;

cl_ulong getStartEndTime(cl_event event) {
	cl_int status;

	cl_ulong start, end;
	status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
	status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);

	return end - start;
}

template <typename Dtype>
class Blob {
public:
	Dtype **data;
	int shape[4];
	int count_with_pad;
	int pad[2];
	int mini_batchs;
	Blob(int m) {
		mini_batchs = m;
		data = new Dtype*[mini_batchs];
	}

#if Fixed_point
	int frac;
#endif

	Blob() {};
	Blob(int Channel, int Height, int Width, int pad_Height = 0, int pad_Width = 0, int zero_layer = 0) {

#if KERNEL_MAIN && on_RD
		if (Channel % KERNEL_PARALLEL != 0) {
			printf("channel should be multiple of KERNEL_PARALLEL, ");
			printf("%d auto padding to %d \n", Channel, Channel + KERNEL_PARALLEL - Channel % KERNEL_PARALLEL);
			Channel = Channel + KERNEL_PARALLEL - Channel % KERNEL_PARALLEL;
		}
#elif KERNEL_MAIN && !on_RD
		if (Channel % KERNEL_PARALLEL != 0 && !zero_layer) {
			printf("channel should be multiple of KERNEL_PARALLEL, ");
			printf("%d auto padding to %d \n", Channel, Channel + KERNEL_PARALLEL - Channel % KERNEL_PARALLEL);
			Channel = Channel + KERNEL_PARALLEL - Channel % KERNEL_PARALLEL;
		}
		if (Channel % FILTER_PARALLEL != 0 && zero_layer) {
			printf("channel should be multiple of FILTER_PARALLEL, ");
			printf("%d auto padding to %d \n", Channel, Channel + FILTER_PARALLEL - Channel % FILTER_PARALLEL);
			Channel = Channel + FILTER_PARALLEL - Channel % FILTER_PARALLEL;
		}
#else
		if (Channel % FILTER_PARALLEL != 0) {
			printf("channel should be multiple of FILTER_PARALLEL, ");
			printf("%d auto padding to %d \n", Channel, Channel + FILTER_PARALLEL - Channel % FILTER_PARALLEL);
			Channel = Channel + FILTER_PARALLEL - Channel % FILTER_PARALLEL;
		}
#endif

		pad[0] = pad_Height;
		pad[1] = pad_Width;
		shape[0] = mini_batch;
		shape[1] = Channel;
		shape[2] = Height + 2 * pad_Height;
		shape[3] = Width + 2 * pad_Width;

#if Fixed_point
		frac = fraction[frac_ptr][0];
		frac_ptr++;
#endif

		data = new Dtype*[mini_batch];
		count_with_pad = (Height + 2 * pad_Height) * (Width + 2 * pad_Width) * Channel;
		for (int i = 0; i < mini_batch; i++) {
			data[i] = new Dtype[count_with_pad];
			for (int j = 0; j < count_with_pad; j++) {
				data[i][j] = 0;
			}
		}
	}

	void get_shuffled_set(Blob<Dtype> &All, int *shuffle_index, int current_index, int total) {

		if (current_index + mini_batchs >= total) {
			shape[0] = total - current_index;
		}
		else {
			shape[0] = mini_batchs;
		}
		shape[1] = All.shape[1];
		shape[2] = All.shape[2];
		shape[3] = All.shape[3];
		count_with_pad = All.count_with_pad;
		pad[0] = All.pad[0];
		pad[1] = All.pad[1];
#if Fixed_point
		frac = All.frac;
#endif
		for (int i = 0; i < shape[0]; i++) {
			data[i] = All.data[shuffle_index[current_index + i]];
		}
	}

	void get_mini_batch(Blob<Dtype> &All) {

		shape[0] = mini_batchs;
		shape[1] = All.shape[1];
		shape[2] = All.shape[2];
		shape[3] = All.shape[3];
		count_with_pad = All.count_with_pad;
		pad[0] = All.pad[0];
		pad[1] = All.pad[1];
#if Fixed_point
		frac = All.frac;
#endif
		for (int i = 0; i < shape[0]; i++) {
			data[i] = All.data[i];
		}
	}
};

template <typename Dtype>
class Blob_maxPool : public Blob<Dtype> {
public:
	int pool[2];
	int stride[2];
	Blob_maxPool(int Channel, int Height, int Width, int pool_h, int pool_w, int stride_h = 1, int stride_w = 1, int pad_Height = 0, int pad_Width = 0) : Blob<Dtype>(Channel, Height, Width, pad_Height, pad_Width) {
		pool[0] = pool_h;
		pool[1] = pool_w;
		stride[0] = stride_h;
		stride[1] = stride_w;

#if Fixed_point
		frac_ptr--;
#endif
	}
};

template <typename Dtype>
class parameter {
public:
	Dtype *W;
	Dtype *b;
	int original_shape[4];
	int shape[4];
	int stride[2];
	int count;
#if Fixed_point
	int frac;
#endif
	parameter(int dims, int prev_dims, int fH, int fW, int stride_H, int stride_W) {
		original_shape[0] = dims;
		original_shape[1] = prev_dims;
		original_shape[2] = fH;
		original_shape[3] = fW;
#if KERNEL_MAIN
		if (dims % KERNEL_PARALLEL != 0) {
			printf("kernel should be multiple of KERNEL_PARALLEL, ");
			printf("%d auto padding to %d \n", dims, dims + KERNEL_PARALLEL - dims % KERNEL_PARALLEL);
			dims = dims + KERNEL_PARALLEL - dims % KERNEL_PARALLEL;
		}
#else
		if (dims % FILTER_PARALLEL != 0) {
			printf("kernel should be multiple of FILTER_PARALLEL, ");
			printf("%d auto padding to %d \n", dims, dims + FILTER_PARALLEL - dims % FILTER_PARALLEL);
			dims = dims + FILTER_PARALLEL - dims % FILTER_PARALLEL;
		}
#endif
		if (prev_dims % FILTER_PARALLEL != 0) {
			printf("filter should be multiple of FILTER_PARALLEL, ");
			printf("%d auto padding to %d \n", prev_dims, prev_dims + FILTER_PARALLEL - prev_dims % FILTER_PARALLEL);
			prev_dims = prev_dims + FILTER_PARALLEL - prev_dims % FILTER_PARALLEL;
		}
		shape[0] = dims;
		shape[1] = prev_dims;
		shape[2] = fH;
		shape[3] = fW;
		stride[0] = stride_H;
		stride[1] = stride_W;
		count = dims*prev_dims*fH*fW;
#if Fixed_point
		frac = fraction[frac_ptr][1];
		frac_ptr++;
#endif
		W = new Dtype[dims*prev_dims*fH*fW];
		b = new Dtype[dims];
		for (int k = 0; k < shape[0]; k ++) {
			for (int f = 0; f < shape[1]; f ++) {
				for (int h = 0; h < shape[2]; h ++) {
					for (int w = 0; w < shape[3]; w ++) {
						W[(( k * shape[1] + f ) * shape[2] + h ) * shape[3] + w] = 0;
					}
				}
			}
			b[k] = 0;
		}
	}
};

template <typename Dtype>
void Convolution(Blob<Dtype> *Z, parameter<Dtype> *W, Blob<Dtype> *Z_prev) {
#if KERNEL_MAIN
	int prev_dims = W->shape[1];
#else
	int prev_dims = Z_prev->shape[1];
#endif
	int prev_Height_pad = Z_prev->shape[2];
	int prev_Width_pad = Z_prev->shape[3];
	int Kernel = W->shape[0];
	int f_Height = W->shape[2];
	int f_Width = W->shape[3];
	int Height_pad = Z->shape[2];
	int Width_pad = Z->shape[3];
	int pad_h = Z->pad[0];
	int pad_w = Z->pad[1];
	int stride_h = W->stride[0];
	int stride_w = W->stride[1];
	int fk_start, fd_start, fh_start;
	int prev_h_start, prev_w_start, prev_slice_c_start, prev_slice_h_start;
	int k_start, h_start;
#if Fixed_point
	MACTYPE Z_data_cache;
	DPTYPE b_cache;
	int frac_in = Z_prev->frac;
	int	frac_out = Z->frac;
	int	frac_w = W->frac;
#else
	Dtype Z_data_cache;
	Dtype b_cache;
#endif

	// CHECK
	if (Height_pad - 2 * pad_h != ((prev_Height_pad - f_Height) / stride_h + 1)) {
		printf("Conv layer doesn't match height \n");
		printf("Z_prev_h_pad:%d, stride:%d \n", prev_Height_pad, stride_h);
		printf("Z_h_pad:%d, pad_h:%d \n", Height_pad, pad_h);
		printf("Filter_Height:%d \n\n", f_Height);
	}
	if (Width_pad - 2 * pad_w != ((prev_Width_pad - f_Width) / stride_w + 1)) {
		printf("Conv layer doesn't match Width \n");
		printf("Z_prev_w_pad:%d, stride:%d \n", prev_Width_pad, stride_w);
		printf("Z_w_pad:%d, pad_w:%d \n", Width_pad, pad_w);
		printf("Filter_Width:%d \n\n", f_Width);
	}

	tic = std::chrono::system_clock::now();

	for (int i = 0; i < mini_batch; i++) {
		for (int k = 0; k < Kernel; k++) {
			b_cache = W->b[k];
			fk_start = k*prev_dims*f_Height*f_Width;
			k_start = k * Height_pad * Width_pad;
			for (int h = pad_h; h < Height_pad - pad_h; h++) {
				h_start = k_start + h * Width_pad;
				prev_h_start = (h - pad_h) * stride_h * prev_Width_pad;
				for (int w = pad_w; w < Width_pad - pad_w; w++) {
					Z_data_cache = 0;
					prev_w_start = prev_h_start + (w - pad_w) * stride_w;
					for (int c = 0; c < prev_dims; c++) {
						fd_start = fk_start + c*f_Height*f_Width;
						prev_slice_c_start = prev_w_start + c * prev_Height_pad * prev_Width_pad;
						for (int fH = 0; fH < f_Height; fH++) {
							fh_start = fd_start + fH * f_Width;
							prev_slice_h_start = prev_slice_c_start + fH * prev_Width_pad;
							for (int fW = 0; fW < f_Width; fW++) {
								Z_data_cache += W->W[fh_start + fW] * Z_prev->data[i][prev_slice_h_start + fW];
							}
						}
					}
#if Fixed_point
					Z_data_cache += b_cache << (frac_w + frac_in - frac_out);
					if (Z_data_cache >(127 << (frac_w + frac_in - frac_out))) {
						//printf("%d \n", ((Z_data_cache >> (frac_w + frac_in - frac_out - 1)) + 0x1) >> 1);
						Z->data[i][h_start + w] = 0x7F;
					}
					else if (Z_data_cache < (-128 << (frac_w + frac_in - frac_out))) {
						//printf("%d \n", ((Z_data_cache >> (frac_w + frac_in - frac_out - 1)) + 0x1) >> 1);
						Z->data[i][h_start + w] = 0x80;
					}
					else {
						Z->data[i][h_start + w] += ((Z_data_cache >> (frac_w + frac_in - frac_out - 1)) + 0x1) >> 1;
					}
#else
					Z->data[i][h_start + w] = Z_data_cache + b_cache;
#endif
				}
			}
		}
	}
	toc = std::chrono::system_clock::now();
	printf("Conv duration: %7.3f ms\n", std::chrono::duration<double>(toc - tic).count() * 1000);
}

template <typename Dtype>
void maxPool(Blob_maxPool<Dtype> *Z, Blob<Dtype> *Z_prev) {
	int prev_dims = Z_prev->shape[1];
	int prev_Height_pad = Z_prev->shape[2];
	int prev_Width_pad = Z_prev->shape[3];

	int prev_pad_h = Z_prev->pad[0];
	int prev_pad_w = Z_prev->pad[1];

	int Height_pad = Z->shape[2];
	int Width_pad = Z->shape[3];
	int pad_h = Z->pad[0];
	int pad_w = Z->pad[1];
	int pool_h = Z->pool[0];
	int pool_w = Z->pool[1];
	int stride_h = Z->stride[0];
	int stride_w = Z->stride[1];
	int prev_c_start, prev_h_start, prev_w_start, prev_slice_h_start;
	int c_start, h_start;
	Dtype temp_prev_slice_max;


	// CHECK
	if (Height_pad - 2 * pad_h != ((prev_Height_pad - pool_h) / stride_h + 1)) {
		printf("Pool layer doesn't match height \n");
		printf("Z_prev_h_pad:%d, prev_pad_h:%d, stride_h:%d \n", prev_Height_pad, prev_pad_h, stride_h);
		printf("Z_h_pad:%d, pad_h:%d \n\n", Height_pad, pad_h);
	}
	if (Width_pad - 2 * pad_w != ((prev_Width_pad - pool_w) / stride_w + 1)) {
		printf("Pool layer doesn't match Width \n");
		printf("Z_prev_w_pad:%d, prev_pad_w:%d, stride_w:%d \n", prev_Width_pad, prev_pad_w, stride_w);
		printf("Z_w_pad:%d, pad_w:%d \n\n", Width_pad, pad_w);
	}

	tic = std::chrono::system_clock::now();
	for (int i = 0; i < mini_batch; i++) {
		for (int c = 0; c < prev_dims; c++) {
			c_start = c * Height_pad * Width_pad;
			prev_c_start = c * prev_Height_pad * prev_Width_pad;
			for (int h = pad_h; h < Height_pad - pad_h; h++) {
				h_start = c_start + h * Width_pad;
				prev_h_start = prev_c_start + (h - pad_h) * stride_h * prev_Width_pad;
				for (int w = pad_w; w < Width_pad - pad_w; w++) {
					prev_w_start = prev_h_start + (w - pad_w) * stride_w;
					temp_prev_slice_max = Z_prev->data[i][prev_w_start];
					for (int fH = 0; fH < pool_h; fH++) {
						prev_slice_h_start = prev_w_start + fH * prev_Width_pad;
						for (int fW = 0; fW < pool_w; fW++) {
							if (Z_prev->data[i][prev_slice_h_start + fW] > temp_prev_slice_max) {
								temp_prev_slice_max = Z_prev->data[i][prev_slice_h_start + fW];
							}
						}
					}
					Z->data[i][h_start + w] = temp_prev_slice_max;
				}
			}
		}
	}
	toc = std::chrono::system_clock::now();
	printf("Pool duration: %7.3f ms\n", std::chrono::duration<double>(toc - tic).count() * 1000);
}

template <typename Dtype>
void relu(Blob<Dtype> *Z_prev) {
	int neuron = Z_prev->count_with_pad;

	tic = std::chrono::system_clock::now();
	for (int i = 0; i < mini_batch; i++) {
		for (int j = 0; j < neuron; j++) {
			if (Z_prev->data[i][j] < 0)
				Z_prev->data[i][j] = 0;
		}
	}
	toc = std::chrono::system_clock::now();
	printf("relu duration: %7.3f ms\n", std::chrono::duration<double>(toc - tic).count() * 1000);
}

template <typename Dtype>
void read_HDF5_4D(Blob<Dtype> *&container, std::string file_path, std::string data_name, int normalize, int pad_h = 0, int pad_w = 0) {
	hid_t file_id, dataset_id, dataspace;
	int status;
	file_id = H5Fopen(file_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
	dataset_id = H5Dopen2(file_id, data_name.c_str(), H5P_DEFAULT);
	dataspace = H5Dget_space(dataset_id);

	hsize_t dim[4];              /* memory space dimensions */
	for (int i = 0; i < 4; i++) {
		dim[i] = 1;
	}
	status = H5Sget_simple_extent_dims(dataspace, dim, NULL);

	int total_size = 1;
	for (int i = 0; i < 4; i++) {
		total_size *= dim[i];
	}
	int *data_flatten;
	data_flatten = new int[total_size];

	status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
		data_flatten);

	container = new Blob<Dtype>(dim[3], dim[1], dim[2], pad_h, pad_w, 1);

	//convert (m,h,w,c) to (m,c,h,w)
	for (int i = 0; i < mini_batch; i++) {
		for (int h = 0; h < dim[1]; h++) {
			for (int w = 0; w < dim[2]; w++) {
				for (int c = 0; c < dim[3]; c++) {
#if Fixed_point
					container->data[i][(c*(dim[1] + 2 * pad_h) + h + pad_h)*(dim[2] + 2 * pad_w) + w + pad_w] = Dtype((float(data_flatten[((i*dim[1] + h)*dim[2] + w)*dim[3] + c]) / normalize) * (1 << container->frac) + 0.5);
#else
					container->data[i][(c*(dim[1] + 2 * pad_h) + h + pad_h)*(dim[2] + 2 * pad_w) + w + pad_w] = Dtype(data_flatten[((i*dim[1] + h)*dim[2] + w)*dim[3] + c]) / normalize;
#endif
				}
			}
		}
	}

	/* Close the dataset. */
	status = H5Dclose(dataset_id);
	/* Close the file. */
	status = H5Fclose(file_id);
	/* Close the dataspace. */
	status = H5Sclose(dataspace);
	delete data_flatten;
}

template <typename Dtype>
DPTYPE* read_HDF5_PARALLEL(Blob<Dtype> *&container, std::string file_path, std::string data_name, int normalize, int pad_h = 0, int pad_w = 0) {
	hid_t file_id, dataset_id, dataspace;
	int status;
	file_id = H5Fopen(file_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
	dataset_id = H5Dopen2(file_id, data_name.c_str(), H5P_DEFAULT);
	dataspace = H5Dget_space(dataset_id);

	hsize_t dim[4];              /* memory space dimensions */
	for (int i = 0; i < 4; i++) {
		dim[i] = 1;
	}
	status = H5Sget_simple_extent_dims(dataspace, dim, NULL);

	int total_size = 1;
	for (int i = 0; i < 4; i++) {
		total_size *= dim[i];
	}
	int *data_flatten;
	data_flatten = new int[total_size];

	status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
		data_flatten);

	container = new Blob<Dtype>(dim[3], dim[1], dim[2], pad_h, pad_w, 1);
	Dtype **temp;
	temp = new Dtype*[mini_batch];
	for (int i = 0; i < mini_batch; i++) {
		temp[i] = new Dtype[container->count_with_pad];
		for (int j = 0; j < container->count_with_pad; j++) {
			temp[i][j] = 0;
		}
	}
	
	//convert (m,h,w,c) to (m,c,h,w)
	for (int i = 0; i < mini_batch; i++) {
		for (int h = 0; h < dim[1]; h++) {
			for (int w = 0; w < dim[2]; w++) {
				for (int c = 0; c < dim[3]; c++) {
#if Fixed_point
					temp[i][(c*(dim[1] + 2 * pad_h) + h + pad_h)*(dim[2] + 2 * pad_w) + w + pad_w] = Dtype((float(data_flatten[((i*dim[1] + h)*dim[2] + w)*dim[3] + c]) / normalize) * (1 << container->frac) + 0.5);
#else
					temp[i][(c*(dim[1] + 2 * pad_h) + h + pad_h)*(dim[2] + 2 * pad_w) + w + pad_w] = Dtype(data_flatten[((i*dim[1] + h)*dim[2] + w)*dim[3] + c]) / normalize;
#endif
				}
			}
		}
	}
	

	// resort from (m,c,h,w) to ( m , c/FILTER_PARALLEL , h , w*FILTER_PARALLEL )
	// if (1,3,3,3) and FILTER_PARALLEL=3, will be (1,1,3,9)
	// 0 1 2     9 10 11     18 19 20      0  9 18 1 10 19 2 11 20
	// 3 4 5    12 13 14     21 22 23   => 3 12 21 4 13 22 5 14 23
	// 6 7 8    15 16 17     24 25 26      6 15 24 7 16 25 8 17 26
	DPTYPE *data = new DPTYPE[mini_batch*container->count_with_pad];
	int shape[4];
	for (int i = 0; i < 4; i++) shape[i] = container->shape[i];

#if on_RD
	for (int i = 0; i < mini_batch; i++) {
		for (int c = 0; c < shape[1] / KERNEL_PARALLEL; c++) {
			for (int h = 0; h < shape[2]; h++) {
				for (int w = 0; w < shape[3]; w++) {
					for (int p = 0; p < KERNEL_PARALLEL; p++) {
						data[i*container->count_with_pad + c * shape[2] * shape[3] * KERNEL_PARALLEL + h * shape[3] * KERNEL_PARALLEL + w * KERNEL_PARALLEL + p] = temp[i][(c * KERNEL_PARALLEL + p) * shape[2] * shape[3] + h * shape[3] + w];
					}
				}
			}
		}
	}
#else
	for (int i = 0; i < mini_batch; i++) {
		for (int c = 0; c < shape[1] / FILTER_PARALLEL; c++) {
			for (int h = 0; h < shape[2]; h++) {
				for (int w = 0; w < shape[3]; w++) {
					for (int p = 0; p < FILTER_PARALLEL; p++) {
						data[i* container->count_with_pad + c * shape[2] * shape[3] * FILTER_PARALLEL + h * shape[3] * FILTER_PARALLEL + w * FILTER_PARALLEL + p] = temp[i][(c * FILTER_PARALLEL + p) * shape[2] * shape[3] + h * shape[3] + w];
					}
				}
			}
		}
	}
#endif

	/* Close the dataset. */
	status = H5Dclose(dataset_id);
	/* Close the file. */
	status = H5Fclose(file_id);
	/* Close the dataspace. */
	status = H5Sclose(dataspace);
	delete data_flatten;
	for (int i = 0; i < mini_batch; i++) {
		delete temp[i];
	}
	delete[]temp;
	return data;
}

template <typename Dtype>
DPTYPE* read_HDF5_FOLD_HW(Blob<Dtype> *&container, std::string file_path, std::string data_name, int normalize, int pad_h, int pad_w, int f_Height, int f_Width, int stride_h, int stride_w) {
	hid_t file_id, dataset_id, dataspace;
	int status;
	file_id = H5Fopen(file_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
	dataset_id = H5Dopen2(file_id, data_name.c_str(), H5P_DEFAULT);
	dataspace = H5Dget_space(dataset_id);

	hsize_t dim[4];              /* memory space dimensions */
	for (int i = 0; i < 4; i++) {
		dim[i] = 1;
	}
	status = H5Sget_simple_extent_dims(dataspace, dim, NULL);

	int total_size = 1;
	for (int i = 0; i < 4; i++) {
		total_size *= dim[i];
	}
	int *data_flatten;
	data_flatten = new int[total_size];

	status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
		data_flatten);

	int m = dim[0];
	int prev_Height = dim[1];
	int prev_Width = dim[2];
	int prev_Channel = dim[3];

	int Height = (prev_Height + 2 * pad_h - f_Height) / stride_h + 1;
	int Width = (prev_Width + 2 * pad_w - f_Width) / stride_w + 1;
	int Channel = prev_Channel * f_Height * f_Width;

	container = new Blob<Dtype>(Channel, Height, Width, 0, 0, 1);

	Dtype **temp;
	temp = new Dtype*[mini_batch];
	for (int i = 0; i < mini_batch; i++) {
		temp[i] = new Dtype[(prev_Height + 2 * pad_h)*(prev_Width + 2 * pad_w)*prev_Channel];
		for (int j = 0; j < (prev_Height + 2 * pad_h)*(prev_Width + 2 * pad_w)*prev_Channel; j++) {
			temp[i][j] = 0;
		}
	}

	//convert (m,h,w,c) to (m,c,h,w)
	for (int i = 0; i < mini_batch; i++) {
		for (int h = 0; h < prev_Height; h++) {
			for (int w = 0; w < prev_Width; w++) {
				for (int c = 0; c < prev_Channel; c++) {
#if Fixed_point
					temp[i][c*(prev_Height + 2 * pad_h)*(prev_Width + 2 * pad_w) + (h + pad_h)*(prev_Width + 2 * pad_w) + w + pad_w] = Dtype((float(data_flatten[i * prev_Height * prev_Width * prev_Channel + h * prev_Width * prev_Channel + w * prev_Channel + c]) / normalize) * (1 << container->frac) + 0.5);
#else
					temp[i][c*(prev_Height + 2 * pad_h)*(prev_Width + 2 * pad_w) + (h + pad_h)*(prev_Width + 2 * pad_w) + w + pad_w] = Dtype(data_flatten[i * prev_Height * prev_Width * prev_Channel + h * prev_Width * prev_Channel + w * prev_Channel + c]) / normalize;
#endif
				}
			}
		}
	}
	DPTYPE *data;
	data = new DPTYPE[mini_batch * container->count_with_pad];

	//img2col
	int prev_dims = prev_Channel;
	int prev_Height_pad = prev_Height + 2 * pad_h;
	int prev_Width_pad = prev_Width + 2 * pad_w;
	int prev_h_start, prev_w_start, prev_slice_c_start, prev_slice_h_start;
	int k_start, h_start;
	// only get mini batches
	for (int i = 0; i < mini_batch; i++) {
		for (int h = 0; h < Height; h++) {
			prev_h_start = h * stride_h * prev_Width_pad;
			for (int w = 0; w < Width; w++) {
				prev_w_start = prev_h_start + w * stride_w;
				for (int c = 0; c < prev_dims; c++) {
					prev_slice_c_start = prev_w_start + c * prev_Height_pad * prev_Width_pad;
					for (int fH = 0; fH < f_Height; fH++) {
						prev_slice_h_start = prev_slice_c_start + fH * prev_Width_pad;
						for (int fW = 0; fW < f_Width; fW++) {
#if on_RD
							data[i*container->count_with_pad + int((((c*f_Height*f_Width) + fH*f_Width + fW) / KERNEL_PARALLEL))*Height*Width*KERNEL_PARALLEL + h*Width*KERNEL_PARALLEL + w*KERNEL_PARALLEL + ((c*f_Height*f_Width) + fH*f_Width + fW) % KERNEL_PARALLEL] = temp[i][prev_slice_h_start + fW];
#else
							data[i*container->count_with_pad + int((((c*f_Height*f_Width) + fH*f_Width + fW) / FILTER_PARALLEL))*Height*Width*FILTER_PARALLEL + h*Width*FILTER_PARALLEL + w*FILTER_PARALLEL + ((c*f_Height*f_Width) + fH*f_Width + fW) % FILTER_PARALLEL] = temp[i][prev_slice_h_start + fW];
#endif
						}
					}
				}
			}
		}
	}


	/* Close the dataset. */
	status = H5Dclose(dataset_id);
	/* Close the file. */
	status = H5Fclose(file_id);
	/* Close the dataspace. */
	status = H5Sclose(dataspace);
	delete data_flatten;
	for (int i = 0; i < mini_batch; i++) {
		delete temp[i];
	}
	delete[]temp;

	return data;
}

template <typename Dtype>
DPTYPE* read_HDF5_FOLD_H(Blob<Dtype> *&container, std::string file_path, std::string data_name, int normalize, int pad_h, int pad_w, int f_Height, int f_Width, int stride_h, int stride_w) {
	hid_t file_id, dataset_id, dataspace;
	int status;
	file_id = H5Fopen(file_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
	dataset_id = H5Dopen2(file_id, data_name.c_str(), H5P_DEFAULT);
	dataspace = H5Dget_space(dataset_id);

	hsize_t dim[4];              /* memory space dimensions */
	for (int i = 0; i < 4; i++) {
		dim[i] = 1;
	}
	status = H5Sget_simple_extent_dims(dataspace, dim, NULL);

	int total_size = 1;
	for (int i = 0; i < 4; i++) {
		total_size *= dim[i];
	}
	int *data_flatten;
	data_flatten = new int[total_size];

	status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
		data_flatten);

	int m = dim[0];
	int prev_Height = dim[1];
	int prev_Width = dim[2];
	int prev_Channel = dim[3];

	int Height = (prev_Height + 2 * pad_h - f_Height) / stride_h + 1;
	int Width = (prev_Width + 2 * pad_w) / stride_w;
	int Channel = prev_Channel * f_Height * stride_w;

	container = new Blob<Dtype>(Channel, Height, Width, 0, 0, 1);

	Dtype **temp;
	temp = new Dtype*[mini_batch];
	for (int i = 0; i < mini_batch; i++) {
		temp[i] = new Dtype[(prev_Height + 2 * pad_h)*(prev_Width + 2 * pad_w)*prev_Channel];
		for (int j = 0; j < (prev_Height + 2 * pad_h)*(prev_Width + 2 * pad_w)*prev_Channel; j++) {
			temp[i][j] = 0;
		}
	}

	
	//convert (m,h,w,c) to (m,c,h,w)
	for (int i = 0; i < mini_batch; i++) {
		for (int h = 0; h < prev_Height; h++) {
			for (int w = 0; w < prev_Width; w++) {
				for (int c = 0; c < prev_Channel; c++) {
#if Fixed_point
					temp[i][c*(prev_Height + 2 * pad_h)*(prev_Width + 2 * pad_w) + (h + pad_h)*(prev_Width + 2 * pad_w) + w + pad_w] = Dtype((float(data_flatten[i * prev_Height * prev_Width * prev_Channel + h * prev_Width * prev_Channel + w * prev_Channel + c]) / normalize) * (1 << container->frac) + 0.5);
#else
					temp[i][c*(prev_Height + 2 * pad_h)*(prev_Width + 2 * pad_w) + (h + pad_h)*(prev_Width + 2 * pad_w) + w + pad_w] = Dtype(data_flatten[i * prev_Height * prev_Width * prev_Channel + h * prev_Width * prev_Channel + w * prev_Channel + c]) / normalize;
#endif
				}
			}
		}
	}
	
	DPTYPE *data;
	data = new DPTYPE[mini_batch * container->count_with_pad];

	//img2col
	int prev_dims = prev_Channel;
	int prev_Height_pad = prev_Height + 2 * pad_h;
	int prev_Width_pad = prev_Width + 2 * pad_w;
	int Width_pad = container->shape[3];
	int Height_pad = container->shape[2];
	int prev_h_start, prev_w_start, prev_slice_c_start, prev_slice_h_start;
	int k_start, h_start;
	// only get mini batches
	for (int i = 0; i < mini_batch; i++) {
		for (int h = 0; h < Height; h++) {
			prev_h_start = h * stride_h * prev_Width_pad;
			for (int w = 0; w < Width_pad; w++) {
				prev_w_start = prev_h_start + w * stride_w;
				for (int c = 0; c < prev_dims; c++) {
					prev_slice_c_start = prev_w_start + c * prev_Height_pad * prev_Width_pad;
					for (int fH = 0; fH < f_Height; fH++) {
						prev_slice_h_start = prev_slice_c_start + fH * prev_Width_pad;
						for (int fW = 0; fW < stride_w; fW++) {
#if on_RD
							data[i*container->count_with_pad + int((((c*f_Height*stride_w) + fH*stride_w + fW) / KERNEL_PARALLEL))*Height*Width_pad*KERNEL_PARALLEL + h*Width_pad*KERNEL_PARALLEL + w*KERNEL_PARALLEL + ((c*f_Height*stride_w) + fH*stride_w + fW) % KERNEL_PARALLEL] = temp[i][prev_slice_h_start + fW];
#else
							data[i*container->count_with_pad + int((((c*f_Height*stride_w) + fH*stride_w + fW) / FILTER_PARALLEL))*Height*Width_pad*FILTER_PARALLEL + h*Width_pad*FILTER_PARALLEL + w*FILTER_PARALLEL + ((c*f_Height*stride_w) + fH*stride_w + fW) % FILTER_PARALLEL] = temp[i][prev_slice_h_start + fW];
#endif
						}
					}
				}
			}
		}
	}


	/* Close the dataset. */
	status = H5Dclose(dataset_id);
	/* Close the file. */
	status = H5Fclose(file_id);
	/* Close the dataspace. */
	status = H5Sclose(dataspace);
	delete data_flatten;
	for (int i = 0; i < mini_batch; i++) {
		delete temp[i];
	}
	delete[]temp;

	return data;
}

template <typename Dtype>
void laod_HDF5_4D(parameter<Dtype> &container, std::string file_path, std::string data_name, int config = 0) {
	hid_t file_id, dataset_id, dataspace;
	int status;
	file_id = H5Fopen(file_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
	dataset_id = H5Dopen2(file_id, data_name.c_str(), H5P_DEFAULT);
	dataspace = H5Dget_space(dataset_id);

	hsize_t dim[4];              /* memory space dimensions */
	for (int i = 0; i < 4; i++) {
		dim[i] = 1;
	}
	status = H5Sget_simple_extent_dims(dataspace, dim, NULL);

	int total_size = 1;
	for (int i = 0; i < 4; i++) {
		total_size *= dim[i];
	}
	float *data_flatten;
	data_flatten = new float[total_size];
	status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_flatten);

	//(k, f, h, w)
	if (config == 1) { // load fully layer
		for (int k = 0; k < dim[0]; k ++) {
			for (int f = 0; f < dim[1] / container.shape[2] / container.shape[3]; f ++) {
				for (int h = 0; h < container.shape[2]; h ++) {
					for (int w = 0; w < container.shape[3]; w ++) {
#if Fixed_point
						if (data_flatten[k * dim[1] * dim[2] * dim[3] + f * container.shape[2] * container.shape[3] + h * container.shape[3] + w] * (1 << container.frac) > DPTYPE_HI) {
							container.W[k * container.shape[1] * container.shape[2] * container.shape[3] + f * container.shape[2] * container.shape[3] + h * container.shape[3] + w] = DPTYPE_HI;
						}
						else if (data_flatten[k * dim[1] * dim[2] * dim[3] + f * container.shape[2] * container.shape[3] + h * container.shape[3] + w] * (1 << container.frac) < Dtype(DPTYPE_LO)) {
							container.W[k * container.shape[1] * container.shape[2] * container.shape[3] + f * container.shape[2] * container.shape[3] + h * container.shape[3] + w] = DPTYPE_LO;
						}
						else {
							container.W[k * container.shape[1] * container.shape[2] * container.shape[3] + f * container.shape[2] * container.shape[3] + h * container.shape[3] + w] = Dtype(round(data_flatten[k * dim[1] * dim[2] * dim[3] + f * container.shape[2] * container.shape[3] + h * container.shape[3] + w] * (1 << container.frac)));
						}
#else
						container.W[k * container.shape[1] * container.shape[2] * container.shape[3] + f * container.shape[2] * container.shape[3] + h * container.shape[3] + w] = data_flatten[k * dim[1] * dim[2] * dim[3] + f * container.shape[2] * container.shape[3] + h * container.shape[3] + w];
#endif
					}
				}
			}
		}
	}
	else if (config == 2) { // load convolution layer
		for (int k = 0; k < dim[0]; k ++) {
			for (int f = 0; f < dim[1]; f ++) {
				for (int h = 0; h < dim[2]; h ++) {
					for (int w = 0; w < dim[3]; w ++) {
#if Fixed_point
						if (data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w] * (1 << container.frac) > DPTYPE_HI) {
							container.W[k * container.shape[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w] = DPTYPE_HI;
						}
						else if (data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w] * (1 << container.frac) < Dtype(DPTYPE_LO)) {
							container.W[k * container.shape[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w] = DPTYPE_LO;
						}
						else {
							container.W[k * container.shape[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w] = Dtype(round(data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w] * (1 << container.frac)));
						}
#else
						container.W[k * container.shape[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w] = data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w];
#endif
					}
				}
			}
		}
	}
	else { // load bias
		for (int k = 0; k < dim[0]; k ++) {
			for (int f = 0; f < dim[1]; f ++) {
				for (int h = 0; h < dim[2]; h ++) {
					for (int w = 0; w < dim[3]; w ++) {
#if Fixed_point
						if (data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h*dim[3] + w] * (1 << container.frac) > DPTYPE_HI) {
							container.b[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w] = DPTYPE_HI;
							printf("fully weight:%f, frac:%d\n", data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w], container.frac);
						}
						else if (data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h*dim[3] + w] * (1 << container.frac) < Dtype(DPTYPE_LO)) {
							container.b[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w] = DPTYPE_LO;
							printf("fully weight:%f, frac:%d\n", data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w], container.frac);
						}
						else {
							container.b[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w] = Dtype(round(data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w] * (1 << container.frac)));
						}
#else
						container.b[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w] = data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w];
#endif
					}
				}
			}
		}
	}

	/* Close the dataset. */
	status = H5Dclose(dataset_id);
	/* Close the file. */
	status = H5Fclose(file_id);
	/* Close the dataspace. */
	status = H5Sclose(dataspace);
	delete data_flatten;
}

template <typename Dtype>
void laod_HDF5_PARALLEL(parameter<Dtype> *container, std::string file_path, std::string data_name, int config = 0) {
	hid_t file_id, dataset_id, dataspace;
	int status;
	file_id = H5Fopen(file_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
	dataset_id = H5Dopen2(file_id, data_name.c_str(), H5P_DEFAULT);
	dataspace = H5Dget_space(dataset_id);

	hsize_t dim[4];              /* memory space dimensions */
	for (int i = 0; i < 4; i++) {
		dim[i] = 1;
	}
	status = H5Sget_simple_extent_dims(dataspace, dim, NULL);

	int total_size = 1;
	for (int i = 0; i < 4; i++) {
		total_size *= dim[i];
	}
	float *data_flatten;
	data_flatten = new float[total_size];
	status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_flatten);


	Dtype *temp;
	int shape[4] = { 1,1,1,1 };
	//(k, f, h, w)
	if (config == 1) { // load fully layer
		for (int i = 0; i < 4; i ++) shape[i] = container->shape[i];
		temp = new Dtype[shape[0] * shape[1] * shape[2] * shape[3]];
		for (int i = 0; i < shape[0] * shape[1] * shape[2] * shape[3]; i ++) temp[i] = 0;
		for (int k = 0; k < dim[0]; k ++) {
			for (int f = 0; f < dim[1] / shape[2] / shape[3]; f ++) {
				for (int h = 0; h < shape[2]; h ++) {
					for (int w = 0; w < shape[3]; w ++) {
#if Fixed_point
						if (data_flatten[k * dim[1] * dim[2] * dim[3] + f * shape[2] * shape[3] + h * shape[3] + w] * (1 << container->frac) > DPTYPE_HI) {
							temp[k * shape[1] * shape[2] * shape[3] + f * shape[2] * shape[3] + h * shape[3] + w] = DPTYPE_HI;
							printf("fully weight:%f, frac:%d\n", data_flatten[ k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w], container->frac);
						}
						else if (data_flatten[k * dim[1] * dim[2] * dim[3] + f * shape[2] * shape[3] + h * shape[3] + w] * (1 << container->frac) < Dtype(DPTYPE_LO)) {
							temp[k * shape[1] * shape[2] * shape[3] + f * shape[2] * shape[3] + h * shape[3] + w] = DPTYPE_LO;
							printf("fully weight:%f, frac:%d\n", data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w], container->frac);
						}
						else {
							temp[k * shape[1] * shape[2] * shape[3] + f * shape[2] * shape[3] + h * shape[3] + w] = Dtype(round(data_flatten[k * dim[1] * dim[2] * dim[3] + f * shape[2] * shape[3] + h * shape[3] + w] * (1 << container->frac)));
						}
#else
						temp[k * shape[1] * shape[2] * shape[3] + f * shape[2] * shape[3] + h * shape[3] + w] = (data_flatten[k * dim[1] * dim[2] * dim[3] + f * shape[2] * shape[3] + h * shape[3] + w]);
#endif
					}
				}
			}
		}
	}
	else if (config == 2) { // load convolution layer
		for (int i = 0; i < 4; i++) shape[i] = container->shape[i];
		temp = new Dtype[shape[0] * shape[1] * shape[2] * shape[3]];
		for (int i = 0; i < shape[0] * shape[1] * shape[2] * shape[3]; i++) temp[i] = 0;
		for (int k = 0; k < dim[0]; k ++) {
			for (int f = 0; f < dim[1]; f ++) {
				for (int h = 0; h < dim[2]; h ++) {
					for (int w = 0; w < dim[3]; w ++) {
#if Fixed_point
						if (data_flatten[ k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w] * (1 << container->frac) > DPTYPE_HI) {
							temp[ k * shape[1] * shape[2] * shape[3] + f * dim[2] * dim[3] + h * dim[3] + w] = DPTYPE_HI;
							printf("conv weight:%f, frac:%d\n", data_flatten[ k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w], container->frac);
						}
						else if (data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w] * (1 << container->frac) < Dtype(DPTYPE_LO)) {
							temp[ k * shape[1] * shape[2] * shape[3] + f * dim[2] * dim[3] + h * dim[3] + w] = DPTYPE_LO;
							printf("conv weight:%f, frac:%d\n", data_flatten[ k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w], container->frac);
						}
						else {
							temp[ k * shape[1] * shape[2] * shape[3] + f * dim[2] * dim[3] + h * dim[3] + w] = Dtype(round(data_flatten[ k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w] * (1 << container->frac)));
						}
#else
						temp[ k * shape[1] * shape[2] * shape[3] + f * dim[2] * dim[3] + h * dim[3] + w] = (data_flatten[ k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w]);
#endif
					}
				}
			}
		}
	}
	else { // load bias
		shape[0] = container->shape[0];
		temp = new Dtype[shape[0]];
		for (int i = 0; i <shape[0]; i++) temp[i] = 0;
		for (int k = 0; k < dim[0]; k ++) {
			for (int f = 0; f < dim[1]; f ++) {
				for (int h = 0; h < dim[2]; h++) {
					for (int w = 0; w < dim[3]; w++) {
#if Fixed_point
						if (data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w] * (1 << container->frac) > DPTYPE_HI) {
							temp[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w] = DPTYPE_HI;
							printf("fully weight:%f, frac:%d\n", data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w], container->frac);
						}
						else if (data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w] * (1 << container->frac) < Dtype(DPTYPE_LO)) {
							temp[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w] = DPTYPE_LO;
							printf("fully weight:%f, frac:%d\n", data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w], container->frac);
						}
						else {
							temp[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w] = Dtype(round(data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w] * (1 << container->frac)));
						}
#else
						temp[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w] = (data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w]);
#endif
					}
				}
			}
		}
	}
	// reorder
	// from (dim,prev_dim,h,w) to (dim/KERNEL_PARALLEL,prev_dim/FILTER_PARALLEL,h,w*KERNEL_PARALLEL*FILTER_PARALLEL)
	if (config == 1 || config == 2) {
		for (int k = 0; k < shape[0] / KERNEL_PARALLEL; k ++) {
			for (int f = 0; f < shape[1] / FILTER_PARALLEL; f ++) {
				for (int h = 0; h < shape[2]; h ++) {
					for (int w = 0; w < shape[3]; w ++) {
						for (int p = 0; p < KERNEL_PARALLEL; p ++) {
							for (int o = 0; o < FILTER_PARALLEL; o ++) {
								container->W[k * shape[1] * shape[2] * shape[3] * KERNEL_PARALLEL + f * shape[2] * shape[3] * FILTER_PARALLEL * KERNEL_PARALLEL + h * shape[3] * FILTER_PARALLEL * KERNEL_PARALLEL + w * FILTER_PARALLEL * KERNEL_PARALLEL + p * FILTER_PARALLEL + o] = temp[(k * KERNEL_PARALLEL + p) * shape[1] * shape[2] * shape[3] + (f * FILTER_PARALLEL + o) * shape[2] * shape[3] + h * shape[3] + w];
							}
						}
					}
				}
			}
		}
	}
	else {
		for (int k = 0; k < shape[0] / KERNEL_PARALLEL; k ++) {
			for (int f = 0; f < shape[1]; f ++) {
				for (int p = 0; p < KERNEL_PARALLEL; p ++) {
					container->b[k * shape[1] * KERNEL_PARALLEL + f * KERNEL_PARALLEL + p] = temp[(k * KERNEL_PARALLEL + p) * shape[1] + f ];
				}
			}
		}
	}

	/* Close the dataset. */
	status = H5Dclose(dataset_id);
	/* Close the file. */
	status = H5Fclose(file_id);
	/* Close the dataspace. */
	status = H5Sclose(dataspace);
	delete data_flatten;
	delete temp;
}

template <typename Dtype>
void laod_HDF5_FOLD(parameter<Dtype> *&container, std::string file_path, std::string data_name, int config = 0) {
	hid_t file_id, dataset_id, dataspace;
	int status;
	file_id = H5Fopen(file_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
	dataset_id = H5Dopen2(file_id, data_name.c_str(), H5P_DEFAULT);
	dataspace = H5Dget_space(dataset_id);

	hsize_t dim[4];              /* memory space dimensions */
	for (int i = 0; i < 4; i++) {
		dim[i] = 1;
	}
	status = H5Sget_simple_extent_dims(dataspace, dim, NULL);

	int total_size = 1;
	for (int i = 0; i < 4; i++) {
		total_size *= dim[i];
	}
	float *data_flatten;
	data_flatten = new float[total_size];
	status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_flatten);

	//folding reshape
	int new_shape[4];
	new_shape[0] = container->original_shape[0];
	new_shape[1] = container->original_shape[1] * container->original_shape[2] * container->stride[1];
	new_shape[2] = 1;
	if (container->original_shape[3] % container->stride[1] == 0) {
		new_shape[3] = container->original_shape[3] / container->stride[1];
	}
	else {
		new_shape[3] = container->original_shape[3] / container->stride[1] + container->stride[1] - container->original_shape[3] % container->stride[1];
	}
	int new_stride[2];
	new_stride[0] = 1;
	new_stride[1] = 1;
	parameter<Dtype> *new_container = new parameter<Dtype>(new_shape[0], new_shape[1], new_shape[2], new_shape[3], new_stride[0], new_stride[1]);
	new_container->frac = container->frac;

	Dtype *temp;
	int shape[4] = { 1,1,1,1 };
	// folding
	if (config == 2) { // load convolution layer
		for (int i = 0; i < 4; i++) shape[i] = new_container->shape[i];
		temp = new Dtype[shape[0] * shape[1] * shape[2] * shape[3]];
		for (int i = 0; i < shape[0] * shape[1] * shape[2] * shape[3]; i++) temp[i] = 0;
		for (int k = 0; k < dim[0]; k ++) {
			for (int w = 0; w < shape[3] * container->stride[1]; w += container->stride[1]) {
				for (int f = 0; f < dim[1]; f ++) {
					for (int h = 0; h < dim[2]; h ++) {
						for (int s = 0; s < container->stride[1]; s ++) {
							if (w + s < dim[3]) {
#if Fixed_point
								if (data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w + s] * (1 << container->frac) > DPTYPE_HI) {
									temp[k * shape[1] * shape[2] * shape[3] + (f * dim[2] * container->stride[1] + h * container->stride[1] + s) * shape[3] + w / container->stride[1]] = DPTYPE_HI;
									printf("conv weight:%f, frac:%d\n", data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w + s], container->frac);
								}
								else if (data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w  + s] * (1 << container->frac) < Dtype(DPTYPE_LO)) {
									temp[k * shape[1] * shape[2] * shape[3] + (f * dim[2] * container->stride[1] + h * container->stride[1] + s) * shape[3] + w / container->stride[1]] = DPTYPE_LO;
									printf("%d %d %d %d %d\n",k ,f ,h ,w ,s );
									printf("conv weight:%f, frac:%d\n", data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w + s], container->frac);
								}
								else {
									temp[k * shape[1] * shape[2] * shape[3] + (f * dim[2] * container->stride[1] + h * container->stride[1] + s) * shape[3] + w / container->stride[1]] = Dtype(round(data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w + s] * (1 << container->frac)));
								}
#else
								temp[k * shape[1] * shape[2] * shape[3] + (f * dim[2] * container->stride[1] + h * container->stride[1] + s) * shape[3] + w / container->stride[1]] = data_flatten[k * dim[1] * dim[2] * dim[3] + f * dim[2] * dim[3] + h * dim[3] + w + s];
#endif
							}
						}
					}
				}
			}
		}
	}
	
	// reorder
	// from (k,f,h,w) to (k/KERNEL_PARALLEL,f/FILTER_PARALLEL,h,w*KERNEL_PARALLEL*FILTER_PARALLEL)
	if (config == 1 || config == 2) {
		for (int k = 0; k < shape[0] / KERNEL_PARALLEL; k ++) {
			for (int f = 0; f < shape[1] / FILTER_PARALLEL; f ++) {
				for (int h = 0; h < shape[2]; h++) {
					for (int w = 0; w < shape[3]; w++) {
						for (int p = 0; p < KERNEL_PARALLEL; p++) {
							for (int o = 0; o < FILTER_PARALLEL; o++) {
								new_container->W[k * shape[1] * shape[2] * shape[3] * KERNEL_PARALLEL + f * shape[2] * shape[3] * FILTER_PARALLEL * KERNEL_PARALLEL + h * shape[3] * FILTER_PARALLEL * KERNEL_PARALLEL + w * FILTER_PARALLEL * KERNEL_PARALLEL + p * FILTER_PARALLEL + o] = temp[(k * KERNEL_PARALLEL + p) * shape[1] * shape[2] * shape[3] + (f * FILTER_PARALLEL + o) * shape[2] * shape[3] + h * shape[3] + w];
							}
						}
					}
				}
			}
		}
	}

	delete container->W;
	delete container->b;
	delete container;
	container = new_container;

	/* Close the dataset. */
	status = H5Dclose(dataset_id);
	/* Close the file. */
	status = H5Fclose(file_id);
	/* Close the dataspace. */
	status = H5Sclose(dataspace);
	delete data_flatten;
	delete temp;
}

template <typename Dtype>
unsigned long convolution_cl(Blob<Dtype> *Z, parameter<Dtype> *W, Blob<Dtype> *Z_prev, cl_command_queue &queue, int relu, cl_event &event, float output_num =1) {

	cl_int err;
	cl_event cur_event;
	unsigned long duration = 0;
	int arg = 0;

	int W_kernel_size = (W->shape[1] / FILTER_PARALLEL) * W->shape[2] * (W->shape[3] / Winograd_Index);
	int Z_h_loop = W_kernel_size * (Z->shape[2] - 2 * Z->pad[0]) * (int((Z->shape[3] - 2 * Z->pad[1])/ output_num + 0.5)) * mini_batch;
	int Z_k_loop = Z_h_loop * (W->shape[0] / KERNEL_PARALLEL);

	int Z_h_loop_sys = (Z->shape[2] - 2 * Z->pad[0]) * (int((Z->shape[3] - 2 * Z->pad[1])/ output_num +0.5)) * mini_batch;
	int Z_k_loop_sys = Z_h_loop_sys * (W->shape[0] / KERNEL_PARALLEL);

	int frac = W->frac + Z_prev->frac - Z->frac;
	// Set the arguments of the kernel
	err = clSetKernelArg(Conv_kernel, arg++, sizeof(cl_int), (void *)&W_kernel_size);
	err = clSetKernelArg(Conv_kernel, arg++, sizeof(cl_int), (void *)&Z_h_loop);
	err = clSetKernelArg(Conv_kernel, arg++, sizeof(cl_int), (void *)&Z_k_loop);
	err = clSetKernelArg(Conv_kernel, arg++, sizeof(cl_int), (void *)&Z_h_loop_sys);
	err = clSetKernelArg(Conv_kernel, arg++, sizeof(cl_int), (void *)&Z_k_loop_sys);
	err = clSetKernelArg(Conv_kernel, arg++, sizeof(cl_int), (void *)&relu);
#if Fixed_point
	err = clSetKernelArg(Conv_kernel, arg++, sizeof(cl_int), (void *)&frac);
#endif
	size_t local[3] = { 1,1,1 };
	size_t global[3] = { 1,1,1 };
	err = clEnqueueNDRangeKernel(queue, Conv_kernel, 3, NULL, global, local, 1, &event, &cur_event);
	if (remote_duration) {
		clFinish(queue);
		duration = getStartEndTime(cur_event);
		clReleaseEvent(cur_event);
		printf("Conv duration: %10.3f ms\n", (float)duration / 1000000);
	}
	return duration;
}

template <typename Dtype>
void weightRD_cl(Blob<Dtype> *Z, parameter<Dtype> *W, Blob<Dtype> *Z_prev, cl_command_queue &queue, cl_mem &W_mem, cl_event &event,float output_num=1) {

	cl_int err;
	int arg = 0;
	int W_kernel_size = (W->shape[1] / FILTER_PARALLEL) * W->shape[2] * (W->shape[3] / Winograd_Index);
	int Z_h_loop = W_kernel_size * (Z->shape[2] - 2 * Z->pad[0]) * (int((Z->shape[3] - 2 * Z->pad[1])/ output_num +0.5)) * mini_batch;
	int Z_k_loop = Z_h_loop * (W->shape[0] / KERNEL_PARALLEL);
	int Weight_size = W->shape[0] * W_kernel_size;
	// Set the arguments of the kernel
	err = clSetKernelArg(weightRD_kernel, arg++, sizeof(cl_mem), (void *)&W_mem);
	err = clSetKernelArg(weightRD_kernel, arg++, sizeof(cl_int), (void *)&W_kernel_size);
	err = clSetKernelArg(weightRD_kernel, arg++, sizeof(cl_int), (void *)&Z_h_loop);
	err = clSetKernelArg(weightRD_kernel, arg++, sizeof(cl_int), (void *)&Z_k_loop);
	err = clSetKernelArg(weightRD_kernel, arg++, sizeof(cl_int), (void *)&Weight_size);
	size_t global[3] = { 1, 1, 1 };
	size_t local[3] = { 1,1,1 };
	err = clEnqueueNDRangeKernel(queue, weightRD_kernel, 3, NULL, global, local, 1, &event, NULL);
}

template <typename Dtype>
void dataRD_cl(Blob<Dtype> *Z, parameter<Dtype> *W, Blob<Dtype> *Z_prev, cl_command_queue &queue, cl_mem &Z_prev_mem, cl_event &event, float output_num =1) {

	cl_int err;
#if KERNEL_MAIN && on_RD
	int filter_w = W->shape[3] * KERNEL_PARALLEL / FILTER_PARALLEL / Winograd_Index;
	int prev_Width_pad = Z_prev->shape[3] * KERNEL_PARALLEL / FILTER_PARALLEL;
	int prev_w_step = W->stride[1] * KERNEL_PARALLEL / FILTER_PARALLEL * output_num;
	int prev_h_step = W->stride[0] * Z_prev->shape[3] * KERNEL_PARALLEL / FILTER_PARALLEL;
	int Z_prev_size = Z_prev->shape[2] * prev_Width_pad * (Z_prev->shape[1] / KERNEL_PARALLEL); // for space
#else
	int filter_w = W->shape[3] / Winograd_Index;
	int prev_Width_pad = Z_prev->shape[3];
	int prev_w_step = W->stride[1] * output_num;
	int prev_h_step = W->stride[0] * Z_prev->shape[3];
	int Z_prev_size = Z_prev->shape[2] * Z_prev->shape[3] * (Z_prev->shape[1] / FILTER_PARALLEL); // for space
#endif
	int Z_prev_img_size = Z_prev->shape[2] * prev_Width_pad;
	int W_filter_size = W->shape[2] * filter_w;
	int W_kernel_size = W_filter_size * (W->shape[1] / FILTER_PARALLEL);
	int Z_w_loop = W_kernel_size * (int((Z->shape[3] - 2 * Z->pad[1]) / output_num + 0.5));
	int Z_h_loop = Z_w_loop * (Z->shape[2] - 2 * Z->pad[0]);
	int Z_batch_loop = Z_h_loop * mini_batch;
	int Z_k_loop = Z_batch_loop * (W->shape[0] / KERNEL_PARALLEL);

	// Set the arguments of the kernel
	int arg = 0;
	err = clSetKernelArg(dataRD_kernel, arg++, sizeof(cl_mem), (void *)&Z_prev_mem);
	err = clSetKernelArg(dataRD_kernel, arg++, sizeof(cl_int), (void *)&filter_w);
	err = clSetKernelArg(dataRD_kernel, arg++, sizeof(cl_int), (void *)&prev_Width_pad);
	err = clSetKernelArg(dataRD_kernel, arg++, sizeof(cl_int), (void *)&Z_prev_img_size);
	err = clSetKernelArg(dataRD_kernel, arg++, sizeof(cl_int), (void *)&Z_prev_size);
	err = clSetKernelArg(dataRD_kernel, arg++, sizeof(cl_int), (void *)&W_filter_size);
	err = clSetKernelArg(dataRD_kernel, arg++, sizeof(cl_int), (void *)&W_kernel_size);
	err = clSetKernelArg(dataRD_kernel, arg++, sizeof(cl_int), (void *)&Z_w_loop);
	err = clSetKernelArg(dataRD_kernel, arg++, sizeof(cl_int), (void *)&Z_h_loop);
	err = clSetKernelArg(dataRD_kernel, arg++, sizeof(cl_int), (void *)&Z_batch_loop);
	err = clSetKernelArg(dataRD_kernel, arg++, sizeof(cl_int), (void *)&Z_k_loop);
	err = clSetKernelArg(dataRD_kernel, arg++, sizeof(cl_int), (void *)&prev_w_step);
	err = clSetKernelArg(dataRD_kernel, arg++, sizeof(cl_int), (void *)&prev_h_step);
	size_t global[3] = { 1, 1, 1 };
	size_t local[3] = { 1,1,1 };
	err = clEnqueueNDRangeKernel(queue, dataRD_kernel, 3, NULL, global, local, 1, &event, NULL);
}

template <typename Dtype>
void dataWR_cl(Blob<Dtype> *Z, parameter<Dtype> *W, Blob<Dtype> *Z_prev, cl_command_queue &queue, cl_mem &Z_mem, cl_event &event, int output_num = 1, Blob_maxPool<Dtype> *Z_next = NULL) {

	cl_int err;

	int Z_w_loop = (Z->shape[3] - 2 * Z->pad[1]);
	int Z_h_loop = Z_w_loop * (Z->shape[2] - 2 * Z->pad[0]);
	int Z_batch_loop = Z_h_loop * mini_batch;
	int Z_k_loop = Z_batch_loop * W->shape[0] / KERNEL_PARALLEL;

	int next_image_group, next_image, next_size;
	int next_pad_h, next_pad_w, next_Width_pad, next_w_step;
	int stride_h, stride_w, pool_h, pool_w;
	int stride_h_init, stride_w_init;


	if (Z_next != NULL) {
		next_image = Z_next->shape[2] * Z_next->shape[3];
#if KERNEL_MAIN && !on_RD
		next_size = next_image * Z_next->shape[1] / FILTER_PARALLEL;
		next_image_group = Z_next->shape[2] * Z_next->shape[3] * KERNEL_PARALLEL / FILTER_PARALLEL;
#else
		next_size = next_image * Z_next->shape[1] / KERNEL_PARALLEL;
		next_image_group = Z_next->shape[2] * Z_next->shape[3];
#endif
		next_pad_h = Z_next->pad[0];
		next_pad_w = Z_next->pad[1];
		next_Width_pad = Z_next->shape[3];
		next_w_step = next_pad_h * Z_next->shape[3];
		stride_h = Z_next->stride[0];
		stride_w = Z_next->stride[1];
		pool_h = Z_next->pool[0];
		pool_w = Z_next->pool[1];
		stride_h_init = stride_h - pool_h;
		stride_w_init = stride_w - pool_w;
	}
	else {
		next_image = Z->shape[2] * Z->shape[3];
#if KERNEL_MAIN && !on_RD
		next_size = next_image * W->shape[0] / FILTER_PARALLEL;
		next_image_group = Z->shape[2] * Z->shape[3] * KERNEL_PARALLEL / FILTER_PARALLEL;
#else
		next_size = next_image * W->shape[0] / KERNEL_PARALLEL;
		next_image_group = Z->shape[2] * Z->shape[3];
#endif

		next_pad_h = Z->pad[0];
		next_pad_w = Z->pad[1];
		next_Width_pad = Z->shape[3];
		next_w_step = next_pad_h * Z->shape[3];
		stride_h = W->stride[0];
		stride_w = W->stride[1];
		pool_h = W->stride[0];
		pool_w = W->stride[1];
		stride_h_init = 0;
		stride_w_init = 0;
	}

	int arg = 0;
	// Set the arguments of the kernel
	err = clSetKernelArg(dataWR_kernel, arg++, sizeof(cl_mem), (void *)&Z_mem);
	err = clSetKernelArg(dataWR_kernel, arg++, sizeof(cl_int), (void *)&Z_w_loop);
	err = clSetKernelArg(dataWR_kernel, arg++, sizeof(cl_int), (void *)&Z_h_loop);
	err = clSetKernelArg(dataWR_kernel, arg++, sizeof(cl_int), (void *)&Z_batch_loop);
	err = clSetKernelArg(dataWR_kernel, arg++, sizeof(cl_int), (void *)&Z_k_loop);

	err = clSetKernelArg(dataWR_kernel, arg++, sizeof(cl_int), (void *)&next_image_group);
	err = clSetKernelArg(dataWR_kernel, arg++, sizeof(cl_int), (void *)&next_image);
	err = clSetKernelArg(dataWR_kernel, arg++, sizeof(cl_int), (void *)&next_size);
	err = clSetKernelArg(dataWR_kernel, arg++, sizeof(cl_int), (void *)&next_pad_h);
	err = clSetKernelArg(dataWR_kernel, arg++, sizeof(cl_int), (void *)&next_pad_w);
	err = clSetKernelArg(dataWR_kernel, arg++, sizeof(cl_int), (void *)&next_Width_pad);
	err = clSetKernelArg(dataWR_kernel, arg++, sizeof(cl_int), (void *)&next_w_step);
	err = clSetKernelArg(dataWR_kernel, arg++, sizeof(cl_int), (void *)&stride_h);
	err = clSetKernelArg(dataWR_kernel, arg++, sizeof(cl_int), (void *)&stride_w);
	err = clSetKernelArg(dataWR_kernel, arg++, sizeof(cl_int), (void *)&pool_h);
	err = clSetKernelArg(dataWR_kernel, arg++, sizeof(cl_int), (void *)&pool_w);
	err = clSetKernelArg(dataWR_kernel, arg++, sizeof(cl_int), (void *)&stride_h_init);
	err = clSetKernelArg(dataWR_kernel, arg++, sizeof(cl_int), (void *)&stride_w_init);
	err = clSetKernelArg(dataWR_kernel, arg++, sizeof(cl_int), (void *)&output_num);
	size_t global[3] = { 1, 1, 1 };
	size_t local[3] = { 1,1,1 };
	err = clEnqueueNDRangeKernel(queue, dataWR_kernel, 3, NULL, global, local, 0, NULL, &event);
}

template <typename Dtype>
void biasRD_cl(Blob<Dtype> *Z, parameter<Dtype> *W, Blob<Dtype> *Z_prev, cl_command_queue &queue, cl_mem &b_mem, cl_event &event) {

	cl_int err;
	int arg = 0;
	int kernel = W->shape[0] / KERNEL_PARALLEL;
	// Set the arguments of the kernel
	err = clSetKernelArg(biasRD_kernel, arg++, sizeof(cl_mem), (void *)&b_mem);
	err = clSetKernelArg(biasRD_kernel, arg++, sizeof(cl_int), (void *)&kernel);//Kernel
	size_t global[3] = { 1, 1, 1 };
	size_t local[3] = { 1,1,1 };
	err = clEnqueueNDRangeKernel(queue, biasRD_kernel, 3, NULL, global, local, 1, &event, NULL);
}

template <typename Dtype>
void filter_reshape_k_w_f(parameter<Dtype> *W) {
	// If Winograd_Index=1, input_w will be the same as output_w

	// padding for filter_w to be multiple of Winograd_Index
	int old_w = W->shape[3];
	if (old_w % Winograd_Index != 0) {
		W->shape[3] = old_w + Winograd_Index - old_w % Winograd_Index;
		W->count = W->shape[0] * W->shape[1] * W->shape[2] * W->shape[3];
	}
	Dtype *new_W = new Dtype[ W->shape[0] * W->shape[1] * W->shape[2] * W->shape[3] ];
	for (int i = 0; i < W->shape[0] * W->shape[1] * W->shape[2] * W->shape[3]; i++) {
		new_W[i] = 0;
	}
	for (int d = 0; d < W->shape[0] / KERNEL_PARALLEL; d++) {
		for (int p_d = 0; p_d < W->shape[1] / FILTER_PARALLEL; p_d++) {
			for (int h = 0; h < W->shape[2]; h++) {
				for (int p = 0; p < KERNEL_PARALLEL; p++) {
					for (int w = 0; w < old_w; w++) {
						for (int o = 0; o < FILTER_PARALLEL; o++) {
							new_W[d * W->shape[1] * W->shape[2] * W->shape[3] * KERNEL_PARALLEL + p_d * W->shape[2] * W->shape[3] * FILTER_PARALLEL * KERNEL_PARALLEL + h * W->shape[3] * FILTER_PARALLEL * KERNEL_PARALLEL + int(w/Winograd_Index)*FILTER_PARALLEL*KERNEL_PARALLEL*Winograd_Index + p * FILTER_PARALLEL * Winograd_Index + (w % Winograd_Index ) * FILTER_PARALLEL + o] = W->W[d * W->shape[1] * W->shape[2] * old_w * KERNEL_PARALLEL + p_d * W->shape[2] * old_w * FILTER_PARALLEL * KERNEL_PARALLEL + h * old_w * FILTER_PARALLEL * KERNEL_PARALLEL + w * FILTER_PARALLEL * KERNEL_PARALLEL + p * FILTER_PARALLEL + o];
						}
					}
				}
			}
		}
	}
	delete W->W;
	W->W = new_W;
}

template <typename Dtype>
void filter_reshape_w_k_f(parameter<Dtype> *W) {
	// If Winograd_Index=1, input_w will be the same as output_w

	// padding for filter_w to be multiple of Winograd_Index
	int old_w = W->shape[3];
	if (old_w % Winograd_Index != 0) {
		W->shape[3] = old_w + Winograd_Index - old_w % Winograd_Index;
		W->count = W->shape[0] * W->shape[1] * W->shape[2] * W->shape[3];
	}

	Dtype *new_W = new Dtype[W->shape[0] * W->shape[1] * W->shape[2] * W->shape[3]];
	for (int i = 0; i < W->shape[0] * W->shape[1] * W->shape[2] * W->shape[3]; i++) {
		new_W[i] = 0;
	}
	for (int d = 0; d < W->shape[0] / KERNEL_PARALLEL; d++) {
		for (int p_d = 0; p_d < W->shape[1] / FILTER_PARALLEL; p_d++) {
			for (int h = 0; h < W->shape[2]; h++) {
				for (int p = 0; p < KERNEL_PARALLEL; p++) {
					for (int w = 0; w < old_w; w++) {
						for (int o = 0; o < FILTER_PARALLEL; o++) {
							new_W[d * W->shape[1] * W->shape[2] * W->shape[3] * KERNEL_PARALLEL + p_d * W->shape[2] * W->shape[3] * FILTER_PARALLEL * KERNEL_PARALLEL + h * W->shape[3] * FILTER_PARALLEL * KERNEL_PARALLEL + p * FILTER_PARALLEL + w * FILTER_PARALLEL * KERNEL_PARALLEL + o] = W->W[d * W->shape[1] * W->shape[2] * old_w * KERNEL_PARALLEL + p_d * W->shape[2] * old_w * FILTER_PARALLEL * KERNEL_PARALLEL + h * old_w * FILTER_PARALLEL * KERNEL_PARALLEL + w * FILTER_PARALLEL * KERNEL_PARALLEL + p * FILTER_PARALLEL + o];
						}
					}
				}
			}
		}
	}
	delete W->W;
	W->W = new_W;
}

void load_picture(unsigned char **picture, std::string file_path, std::string data_name) {
	hid_t file_id, dataset_id, dataspace;
	int status;
	file_id = H5Fopen(file_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
	dataset_id = H5Dopen2(file_id, data_name.c_str(), H5P_DEFAULT);
	dataspace = H5Dget_space(dataset_id);

	hsize_t dim[4];              /* memory space dimensions */
	for (int i = 0; i < 4; i++) {
		dim[i] = 1;
	}
	status = H5Sget_simple_extent_dims(dataspace, dim, NULL);

	int total_size = 1;
	for (int i = 0; i < 4; i++) {
		total_size *= dim[i];
	}
	int *data_flatten;
	data_flatten = new int[total_size];

	status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
		data_flatten);

	//RGB to BGR for openCV
	for (int i = 0; i < mini_batch; i++) {
		for (int h = 0; h < dim[1]; h++) {
			for (int w = 0; w < dim[2]; w++) {
				for (int c = 0; c < dim[3]; c++) {
					picture[i][ ( h * dim[2] + w ) * dim[3] + dim[3] - c - 1] = data_flatten[((i*dim[1] + h)*dim[2] + w)*dim[3] + c];
				}
			}
		}
	}

	/* Close the dataset. */
	status = H5Dclose(dataset_id);
	/* Close the file. */
	status = H5Fclose(file_id);
	/* Close the dataspace. */
	status = H5Sclose(dataspace);
	delete data_flatten;
}

int runOpenCL(int USE_platform) {
	cl_int err;
	size_t name_size;
	// Get platform information
	cl_uint num_platforms;
	std::string platform_name;
	err = clGetPlatformIDs(0, NULL, &num_platforms);
	printf("\n-----number of platform is %d-----\n", num_platforms);
	platforms = new cl_platform_id[num_platforms];
	err = clGetPlatformIDs(num_platforms, platforms, NULL);
	if (err != CL_SUCCESS) { printf("unable to get PlatformID \n"); }
	for (int i = 0; i < int(num_platforms); i++) {
		err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &name_size);
		platform_name.resize(name_size);
		err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, name_size, &platform_name[0], 0);
		printf("platform %d: %s \n", i, platform_name.c_str());
	}

	printf("-----USE PLATFORM: %d \n", USE_platform);
	// Get device information
	cl_uint num_device;
	std::string device_name;
	err = clGetDeviceIDs(platforms[USE_platform], CL_DEVICE_TYPE_ALL, 0, NULL, &num_device);
	printf("\n-----number of device is %d-----\n", num_device);
	devices = new cl_device_id[num_device];
	err = clGetDeviceIDs(platforms[USE_platform], CL_DEVICE_TYPE_ALL, num_device, devices, NULL);
	if (err != CL_SUCCESS) { printf("unable to get DeviceID \n"); }
	for (int i = 0; i < num_device; i++) {
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &name_size);
		device_name.resize(name_size);
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, name_size, &device_name[0], 0);
		printf("devices %d: %s \n", i, device_name.c_str());
	}

	// Build device
	printf("\n-----Build Program to device \n");
	command_queue = new cl_command_queue[num_device];
	weightRD_queue = new cl_command_queue[num_device];
	biasRD_queue = new cl_command_queue[num_device];
	dataRD_queue = new cl_command_queue[num_device];
	dataWR_queue = new cl_command_queue[num_device];
	program = new cl_program[num_device];

	char *buf = NULL;
	size_t nb;
	context = clCreateContext(NULL, num_device, devices, NULL, NULL, &err);
	if (err != CL_SUCCESS) { printf("unable to create Context\n"); }
	for (int current_device = 0; current_device < num_device; current_device++) {
		command_queue[current_device] = clCreateCommandQueue(context, devices[current_device], CL_QUEUE_PROFILING_ENABLE, &err);
		if (err != CL_SUCCESS) { printf("unable to create CommandQueue \n"); }
		weightRD_queue[current_device] = clCreateCommandQueue(context, devices[current_device], CL_QUEUE_PROFILING_ENABLE, &err);
		if (err != CL_SUCCESS) { printf("unable to create weightRD_queue \n"); }
		biasRD_queue[current_device] = clCreateCommandQueue(context, devices[current_device], CL_QUEUE_PROFILING_ENABLE, &err);
		if (err != CL_SUCCESS) { printf("unable to create biasRD_queue \n"); }
		dataRD_queue[current_device] = clCreateCommandQueue(context, devices[current_device], CL_QUEUE_PROFILING_ENABLE, &err);
		if (err != CL_SUCCESS) { printf("unable to create dataRD_queue \n"); }
		dataWR_queue[current_device] = clCreateCommandQueue(context, devices[current_device], CL_QUEUE_PROFILING_ENABLE, &err);
		if (err != CL_SUCCESS) { printf("unable to create dataWR_queue \n"); }
		if (current_device == 0) {
			std::ifstream bin_file("CNN.aocx", std::ifstream::binary);
			bin_file.seekg(0, bin_file.end);
			nb = bin_file.tellg();
			bin_file.seekg(0, bin_file.beg);
			buf = new char[nb];
			bin_file.read(buf, nb);
		}
		// Creating Program from Binary File
		program[current_device] = clCreateProgramWithBinary(context, 1, &devices[current_device], &nb, (const unsigned char **)&buf, NULL, &err);
		if (err != CL_SUCCESS) { printf(" unable to Build Program with Binary, err=%d\n", err); }
	}
	if (buf != NULL) {
		delete buf;
	}

	return num_device;
}

int main(void) {

	num_devices = runOpenCL(platform);

    frac_ptr = 0;
    printf("load Data\n");
    Blob<DPTYPE> *X;
    DPTYPE *data;

    Blob<DPTYPE> *Z0 = new Blob<DPTYPE>(mini_batch);
    Blob<int> *mini_Y = new Blob<int>(mini_batch);

    Blob<DPTYPE> *Z1_conv = new Blob<DPTYPE>(96, 55, 55);
    Blob_maxPool<DPTYPE> *Z1_pool = new Blob_maxPool<DPTYPE>(96, 27, 27, 3, 3, 2, 2, 2, 2);
    Blob<DPTYPE> *Z2_conv = new Blob<DPTYPE>(256, 27, 27);
    Blob_maxPool<DPTYPE> *Z2_pool = new Blob_maxPool<DPTYPE>(256, 13, 13, 3, 3, 2, 2, 1, 1);
    Blob<DPTYPE> *Z3_conv = new Blob<DPTYPE>(384, 13, 13, 1, 1);
    Blob<DPTYPE> *Z4_conv = new Blob<DPTYPE>(384, 13, 13, 1, 1);
    Blob<DPTYPE> *Z5_conv = new Blob<DPTYPE>(256, 13, 13);
    Blob_maxPool<DPTYPE> *Z5_pool = new Blob_maxPool<DPTYPE>(256, 6, 6, 3, 3, 2, 2);
    Blob<DPTYPE> *Z6 = new Blob<DPTYPE>(4096, 1, 1);
    Blob<DPTYPE> *Z7 = new Blob<DPTYPE>(4096, 1, 1);
    Blob<DPTYPE> **Z8 = new Blob<DPTYPE> *[num_devices];
    for (int current_device = 0; current_device < num_devices; current_device++) {
        Z8[current_device] = new Blob<DPTYPE>(6, 1, 1);
        frac_ptr--;
    }

    frac_ptr = 0;
    printf("load Weight\n");

	// folding the first layer to increase the number of filter for parallelism
    if (1) {
		data = read_HDF5_FOLD_H<DPTYPE>(X, "signs_train.h5", "train_set_x",/*normalize*/255,/*pad*/ 82, 82, 11, 11, 4, 4);
        parameter<DPTYPE> *W1_conv = new parameter<DPTYPE>(96, 3 , 11, 11, 4, 4);
        laod_HDF5_FOLD(W1_conv, "Alexnet.h5", "W1", 2);
    } else {
		data = read_HDF5_FOLD_HW<DPTYPE>(X, "signs_train.h5", "train_set_x",/*normalize*/255,/*pad*/82, 82, 11, 11, 4, 4);
        parameter<DPTYPE> *W1_conv = new parameter<DPTYPE>(96, 3 * 11 * 11, 1, 1, 1, 1);
        laod_HDF5_PARALLEL(W1_conv, "Alexnet.h5", "W1", 2);
    }

    parameter<DPTYPE> *W2_conv = new parameter<DPTYPE>(256, 96, 5, 5, 1, 1);
    laod_HDF5_PARALLEL(W2_conv, "Alexnet.h5", "W2", 2);

    parameter<DPTYPE> *W3_conv = new parameter<DPTYPE>(384, 256, 3, 3, 1, 1);
    laod_HDF5_PARALLEL(W3_conv, "Alexnet.h5", "W3", 2);

    parameter<DPTYPE> *W4_conv = new parameter<DPTYPE>(384, 384, 3, 3, 1, 1);
    laod_HDF5_PARALLEL(W4_conv, "Alexnet.h5", "W4", 2);

    parameter<DPTYPE> *W5_conv = new parameter<DPTYPE>(256, 384, 3, 3, 1, 1);
    laod_HDF5_PARALLEL(W5_conv, "Alexnet.h5", "W5", 2);

    parameter<DPTYPE> *W6 = new parameter<DPTYPE>(4096, 256, 6, 6, 1, 1);
    laod_HDF5_PARALLEL(W6, "Alexnet.h5", "fc6/weights", 1);
    laod_HDF5_PARALLEL(W6, "Alexnet.h5", "fc6/biases", 0);

    parameter<DPTYPE> *W7 = new parameter<DPTYPE>(4096, 4096, 1, 1, 1, 1);
    laod_HDF5_PARALLEL(W7, "Alexnet.h5", "fc7/weights", 1);
    laod_HDF5_PARALLEL(W7, "Alexnet.h5", "fc7/biases", 0);

    parameter<DPTYPE> *W8 = new parameter<DPTYPE>(6, 4096, 1, 1, 1, 1);
    laod_HDF5_PARALLEL(W8, "Alexnet.h5", "fc8/weights", 1);
    laod_HDF5_PARALLEL(W8, "Alexnet.h5", "fc8/biases", 0);

    filter_reshape_k_w_f(W1_conv);
    filter_reshape_k_w_f(W2_conv);
    filter_reshape_k_w_f(W3_conv);
    filter_reshape_k_w_f(W4_conv);
    filter_reshape_k_w_f(W5_conv);
    filter_reshape_k_w_f(W6);
    filter_reshape_k_w_f(W7);
    filter_reshape_k_w_f(W8);

    // Layer 0
    Blob<int> *Y;
    read_HDF5_4D<int>(Y, "signs_train.h5", "train_set_y", 1);
    mini_Y->get_mini_batch(*Y);
    Z0->get_mini_batch(*X);

    Z0_mem = new cl_mem[num_devices];
    W1_conv_mem = new cl_mem[num_devices];
    b1_conv_mem = new cl_mem[num_devices];
    Z1_pool_mem = new cl_mem[num_devices];
    W2_conv_mem = new cl_mem[num_devices];
    b2_conv_mem = new cl_mem[num_devices];
    Z2_pool_mem = new cl_mem[num_devices];
    W3_conv_mem = new cl_mem[num_devices];
    b3_conv_mem = new cl_mem[num_devices];
    Z3_conv_mem = new cl_mem[num_devices];
    W4_conv_mem = new cl_mem[num_devices];
    b4_conv_mem = new cl_mem[num_devices];
    Z4_conv_mem = new cl_mem[num_devices];
    W5_conv_mem = new cl_mem[num_devices];
    b5_conv_mem = new cl_mem[num_devices];
    Z5_pool_mem = new cl_mem[num_devices];
    W6_mem = new cl_mem[num_devices];
    b6_mem = new cl_mem[num_devices];
    Z6_mem = new cl_mem[num_devices];
    W7_mem = new cl_mem[num_devices];
    b7_mem = new cl_mem[num_devices];
    Z7_mem = new cl_mem[num_devices];
    W8_mem = new cl_mem[num_devices];
    b8_mem = new cl_mem[num_devices];
    Z8_mem = new cl_mem[num_devices];

    cl_int err;
    printf("\nloading parameter...\n");
    for (int current_device = 0; current_device < num_devices; current_device++) {
        W1_conv_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_ONLY, W1_conv->count * sizeof(DPTYPE), NULL, &err);
        b1_conv_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_ONLY, W1_conv->shape[0] * sizeof(DPTYPE), NULL, &err);
        W2_conv_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_ONLY, W2_conv->count * sizeof(DPTYPE), NULL, &err);
        b2_conv_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_ONLY, W2_conv->shape[0] * sizeof(DPTYPE), NULL, &err);
        W3_conv_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_ONLY, W3_conv->count * sizeof(DPTYPE), NULL, &err);
        b3_conv_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_ONLY, W3_conv->shape[0] * sizeof(DPTYPE), NULL, &err);
        W4_conv_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_ONLY, W4_conv->count * sizeof(DPTYPE), NULL, &err);
        b4_conv_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_ONLY, W4_conv->shape[0] * sizeof(DPTYPE), NULL, &err);
        W5_conv_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_ONLY, W5_conv->count * sizeof(DPTYPE), NULL, &err);
        b5_conv_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_ONLY, W5_conv->shape[0] * sizeof(DPTYPE), NULL, &err);
        W6_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_ONLY, W6->count * sizeof(DPTYPE), NULL, &err);
        b6_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_ONLY, W6->shape[0] * sizeof(DPTYPE), NULL, &err);
        W7_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_ONLY, W7->count * sizeof(DPTYPE), NULL, &err);
        b7_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_ONLY, W7->shape[0] * sizeof(DPTYPE), NULL, &err);
        W8_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_ONLY, W8->count * sizeof(DPTYPE), NULL, &err);
        b8_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_ONLY, W8->shape[0] * sizeof(DPTYPE), NULL, &err);

        err = clEnqueueWriteBuffer(command_queue[current_device], W1_conv_mem[current_device], CL_FALSE, 0, W1_conv->count * sizeof(DPTYPE), W1_conv->W, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(command_queue[current_device], b1_conv_mem[current_device], CL_FALSE, 0, W1_conv->shape[0] * sizeof(DPTYPE), W1_conv->b, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(command_queue[current_device], W2_conv_mem[current_device], CL_FALSE, 0, W2_conv->count * sizeof(DPTYPE), W2_conv->W, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(command_queue[current_device], b2_conv_mem[current_device], CL_FALSE, 0, W2_conv->shape[0] * sizeof(DPTYPE), W2_conv->b, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(command_queue[current_device], W3_conv_mem[current_device], CL_FALSE, 0, W3_conv->count * sizeof(DPTYPE), W3_conv->W, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(command_queue[current_device], b3_conv_mem[current_device], CL_FALSE, 0, W3_conv->shape[0] * sizeof(DPTYPE), W3_conv->b, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(command_queue[current_device], W4_conv_mem[current_device], CL_FALSE, 0, W4_conv->count * sizeof(DPTYPE), W4_conv->W, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(command_queue[current_device], b4_conv_mem[current_device], CL_FALSE, 0, W4_conv->shape[0] * sizeof(DPTYPE), W4_conv->b, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(command_queue[current_device], W5_conv_mem[current_device], CL_FALSE, 0, W5_conv->count * sizeof(DPTYPE), W5_conv->W, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(command_queue[current_device], b5_conv_mem[current_device], CL_FALSE, 0, W5_conv->shape[0] * sizeof(DPTYPE), W5_conv->b, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(command_queue[current_device], W6_mem[current_device], CL_FALSE, 0, W6->count * sizeof(DPTYPE), W6->W, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(command_queue[current_device], b6_mem[current_device], CL_FALSE, 0, W6->shape[0] * sizeof(DPTYPE), W6->b, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(command_queue[current_device], W7_mem[current_device], CL_FALSE, 0, W7->count * sizeof(DPTYPE), W7->W, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(command_queue[current_device], b7_mem[current_device], CL_FALSE, 0, W7->shape[0] * sizeof(DPTYPE), W7->b, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(command_queue[current_device], W8_mem[current_device], CL_FALSE, 0, W8->count * sizeof(DPTYPE), W8->W, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(command_queue[current_device], b8_mem[current_device], CL_FALSE, 0, W8->shape[0] * sizeof(DPTYPE), W8->b, 0, NULL, NULL);

        Z0_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_WRITE, mini_batch * Z0->count_with_pad * sizeof(DPTYPE), NULL, &err);
        Z1_pool_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_WRITE, mini_batch * Z1_pool->count_with_pad * sizeof(DPTYPE), NULL, &err);
        Z2_pool_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_WRITE, mini_batch * Z2_pool->count_with_pad * sizeof(DPTYPE), NULL, &err);
        Z3_conv_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_WRITE, mini_batch * Z3_conv->count_with_pad * sizeof(DPTYPE), NULL, &err);
        Z4_conv_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_WRITE, mini_batch * Z4_conv->count_with_pad * sizeof(DPTYPE), NULL, &err);
        Z5_pool_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_WRITE, mini_batch * Z5_pool->count_with_pad * sizeof(DPTYPE), NULL, &err);
        Z6_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_WRITE, mini_batch * Z6->count_with_pad * sizeof(DPTYPE), NULL, &err);
        Z7_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_WRITE, mini_batch * Z7->count_with_pad * sizeof(DPTYPE), NULL, &err);
        Z8_mem[current_device] = clCreateBuffer(context, CL_MEM_READ_WRITE, mini_batch * Z8[current_device]->count_with_pad * sizeof(DPTYPE), NULL, &err);

        for (int i = 0; i < mini_batch; i++) {
            err = clEnqueueWriteBuffer(command_queue[current_device], Z1_pool_mem[current_device], CL_FALSE, i * Z1_pool->count_with_pad * sizeof(DPTYPE), Z1_pool->count_with_pad * sizeof(DPTYPE), Z1_pool->data[i], 0, NULL, NULL);
            err = clEnqueueWriteBuffer(command_queue[current_device], Z2_pool_mem[current_device], CL_FALSE, i * Z2_pool->count_with_pad * sizeof(DPTYPE), Z2_pool->count_with_pad * sizeof(DPTYPE), Z2_pool->data[i], 0, NULL, NULL);
            err = clEnqueueWriteBuffer(command_queue[current_device], Z3_conv_mem[current_device], CL_FALSE, i * Z3_conv->count_with_pad * sizeof(DPTYPE), Z3_conv->count_with_pad * sizeof(DPTYPE), Z3_conv->data[i], 0, NULL, NULL);
            err = clEnqueueWriteBuffer(command_queue[current_device], Z4_conv_mem[current_device], CL_FALSE, i * Z4_conv->count_with_pad * sizeof(DPTYPE), Z4_conv->count_with_pad * sizeof(DPTYPE), Z4_conv->data[i], 0, NULL, NULL);
            err = clEnqueueWriteBuffer(command_queue[current_device], Z5_pool_mem[current_device], CL_FALSE, i * Z5_pool->count_with_pad * sizeof(DPTYPE), Z5_pool->count_with_pad * sizeof(DPTYPE), Z5_pool->data[i], 0, NULL, NULL);
            err = clEnqueueWriteBuffer(command_queue[current_device], Z6_mem[current_device], CL_FALSE, i * Z6->count_with_pad * sizeof(DPTYPE), Z6->count_with_pad * sizeof(DPTYPE), Z6->data[i], 0, NULL, NULL);
            err = clEnqueueWriteBuffer(command_queue[current_device], Z7_mem[current_device], CL_FALSE, i * Z7->count_with_pad * sizeof(DPTYPE), Z7->count_with_pad * sizeof(DPTYPE), Z7->data[i], 0, NULL, NULL);
            err = clEnqueueWriteBuffer(command_queue[current_device], Z8_mem[current_device], CL_FALSE, i * Z8[current_device]->count_with_pad * sizeof(DPTYPE), Z8[current_device]->count_with_pad * sizeof(DPTYPE), Z8[current_device]->data[i], 0, NULL, NULL);
        }
    }

    printf("Waiting for loading parameter finish...\n");
    for (int current_device = 0; current_device < num_devices; current_device++) {
        tic = std::chrono::system_clock::now();
        clFinish(command_queue[current_device]);
        toc = std::chrono::system_clock::now();
        printf("Waiting on device %d for %7.3f ms\n", current_device, std::chrono::duration<double>(toc - tic).count() * 1000);
    }

    printf("\nloading DEVICE\n");
    for (int current_device = 0; current_device < num_devices; current_device++) {
        printf("-----load DEVICE:%d start-----\n", current_device);
        // Create the OpenCL kernel
        Conv_kernel = clCreateKernel(program[current_device], "conv_core", &err);
        weightRD_kernel = clCreateKernel(program[current_device], "weightRD", &err);
        biasRD_kernel = clCreateKernel(program[current_device], "biasRD", &err);
        dataRD_kernel = clCreateKernel(program[current_device], "dataRD", &err);
        dataWR_kernel = clCreateKernel(program[current_device], "dataWR", &err);

        tic2 = std::chrono::system_clock::now();

        // Layer 0
        err = clEnqueueWriteBuffer(dataRD_queue[current_device], Z0_mem[current_device], CL_FALSE, 0, mini_batch * Z0->count_with_pad * sizeof(DPTYPE), data, 0, NULL, &event[current_device][0]);
        if (remote_duration) {
            tic = std::chrono::system_clock::now();
            clFinish(dataRD_queue[current_device]);
            toc = std::chrono::system_clock::now();
            printf("data transfer: %10.3f ms\n", std::chrono::duration<double>(toc - tic).count() * 1000);
        }
        // Layer 1
        // conv
        dataRD_cl(Z1_conv, W1_conv, Z0, dataRD_queue[current_device], Z0_mem[current_device], event[current_device][0], Winograd_output);
        weightRD_cl(Z1_conv, W1_conv, Z0, weightRD_queue[current_device], W1_conv_mem[current_device], event[current_device][0], Winograd_output);
        biasRD_cl(Z1_conv, W1_conv, Z0, biasRD_queue[current_device], b1_conv_mem[current_device], event[current_device][0]);
        dataWR_cl(Z1_conv, W1_conv, Z0, dataWR_queue[current_device], Z1_pool_mem[current_device], event[current_device][1], Winograd_output, Z1_pool);
        convolution_cl(Z1_conv, W1_conv, Z0, command_queue[current_device], use_relu, event[current_device][0], Winograd_output);
        
        // Layer 2
        // conv
        dataRD_cl(Z2_conv, W2_conv, Z1_pool, dataRD_queue[current_device], Z1_pool_mem[current_device], event[current_device][1], Winograd_output);
        weightRD_cl(Z2_conv, W2_conv, Z1_pool, weightRD_queue[current_device], W2_conv_mem[current_device], event[current_device][1], Winograd_output);
        biasRD_cl(Z2_conv, W2_conv, Z1_pool, biasRD_queue[current_device], b2_conv_mem[current_device], event[current_device][1]);
        dataWR_cl(Z2_conv, W2_conv, Z1_pool, dataWR_queue[current_device], Z2_pool_mem[current_device], event[current_device][2], Winograd_output,Z2_pool);
        convolution_cl(Z2_conv, W2_conv, Z1_pool, command_queue[current_device], use_relu, event[current_device][1], Winograd_output);
        
        // Layer 3
        // conv
        dataRD_cl(Z3_conv, W3_conv, Z2_pool, dataRD_queue[current_device], Z2_pool_mem[current_device], event[current_device][2], Winograd_output);
        weightRD_cl(Z3_conv, W3_conv, Z2_pool, weightRD_queue[current_device], W3_conv_mem[current_device], event[current_device][2], Winograd_output);
        biasRD_cl(Z3_conv, W3_conv, Z2_pool, biasRD_queue[current_device], b3_conv_mem[current_device], event[current_device][2]);
        dataWR_cl(Z3_conv, W3_conv, Z2_pool, dataWR_queue[current_device], Z3_conv_mem[current_device], event[current_device][3], Winograd_output);
        convolution_cl(Z3_conv, W3_conv, Z2_pool, command_queue[current_device], use_relu, event[current_device][2], Winograd_output);

        // Layer 4
        // conv
        dataRD_cl(Z4_conv, W4_conv, Z3_conv, dataRD_queue[current_device], Z3_conv_mem[current_device], event[current_device][3], Winograd_output);
        weightRD_cl(Z4_conv, W4_conv, Z3_conv, weightRD_queue[current_device], W4_conv_mem[current_device], event[current_device][3], Winograd_output);
        biasRD_cl(Z4_conv, W4_conv, Z3_conv, biasRD_queue[current_device], b4_conv_mem[current_device], event[current_device][3]);
        dataWR_cl(Z4_conv, W4_conv, Z3_conv, dataWR_queue[current_device], Z4_conv_mem[current_device], event[current_device][4], Winograd_output);
        convolution_cl(Z4_conv, W4_conv, Z3_conv, command_queue[current_device], use_relu, event[current_device][3], Winograd_output);

        // Layer 5
        // conv
        dataRD_cl(Z5_conv, W5_conv, Z4_conv, dataRD_queue[current_device], Z4_conv_mem[current_device], event[current_device][4], Winograd_output);
        weightRD_cl(Z5_conv, W5_conv, Z4_conv, weightRD_queue[current_device], W5_conv_mem[current_device], event[current_device][4], Winograd_output);
        biasRD_cl(Z5_conv, W5_conv, Z4_conv, biasRD_queue[current_device], b5_conv_mem[current_device], event[current_device][4]);
        dataWR_cl(Z5_conv, W5_conv, Z4_conv, dataWR_queue[current_device], Z5_pool_mem[current_device], event[current_device][5], Winograd_output,Z5_pool);
        convolution_cl(Z5_conv, W5_conv,Z4_conv, command_queue[current_device], use_relu, event[current_device][4], Winograd_output);

        // Layer 6
        //FC
        dataRD_cl(Z6, W6, Z5_pool, dataRD_queue[current_device], Z5_pool_mem[current_device], event[current_device][5]);
        weightRD_cl(Z6, W6, Z5_pool, weightRD_queue[current_device], W6_mem[current_device], event[current_device][5]);
        biasRD_cl(Z6, W6, Z5_pool, biasRD_queue[current_device], b6_mem[current_device], event[current_device][5]);
        dataWR_cl(Z6, W6, Z5_pool, dataWR_queue[current_device], Z6_mem[current_device], event[current_device][6]);
        convolution_cl(Z6, W6, Z5_pool, command_queue[current_device], use_relu, event[current_device][5]);

        // Layer 7
        // FC
        dataRD_cl(Z7, W7, Z6, dataRD_queue[current_device], Z6_mem[current_device], event[current_device][6]);
        weightRD_cl(Z7, W7, Z6, weightRD_queue[current_device], W7_mem[current_device], event[current_device][6]);
        biasRD_cl(Z7, W7, Z6, biasRD_queue[current_device], b7_mem[current_device], event[current_device][6]);
        dataWR_cl(Z7, W7, Z6, dataWR_queue[current_device], Z7_mem[current_device], event[current_device][7]);
        convolution_cl(Z7, W7, Z6, command_queue[current_device], use_relu, event[current_device][6]);

        // Layer 8
        // FC
        dataRD_cl(Z8[current_device], W8, Z7, dataRD_queue[current_device], Z7_mem[current_device], event[current_device][7]);
        weightRD_cl(Z8[current_device], W8, Z7, weightRD_queue[current_device], W8_mem[current_device], event[current_device][7]);
        biasRD_cl(Z8[current_device], W8, Z7, biasRD_queue[current_device], b8_mem[current_device], event[current_device][7]);
        dataWR_cl(Z8[current_device], W8, Z7, dataWR_queue[current_device], Z8_mem[current_device], event[current_device][8]);
        convolution_cl(Z8[current_device], W8, Z7, command_queue[current_device], !use_relu, event[current_device][7]);
        
        // Read the memory buffer 
        for (int i = 0; i < mini_batch; i++) {
            err = clEnqueueReadBuffer(dataWR_queue[current_device], Z8_mem[current_device], CL_FALSE, i * Z8[current_device]->count_with_pad * sizeof(DPTYPE), Z8[current_device]->count_with_pad * sizeof(DPTYPE), Z8[current_device]->data[i], 1, &event[current_device][8], NULL);
        }
        toc2 = std::chrono::system_clock::now();
        if (remote_duration) {
            printf("Device:%d duration: %7.3f ms\n\n", current_device, std::chrono::duration<double>(toc2 - tic2).count() * 1000);
        }
    }

    tic = std::chrono::system_clock::now();
    printf("Waiting for work to finish\n");
    for (int current_device = 0; current_device < num_devices; current_device++) {
        clFinish(dataWR_queue[current_device]);
    }
    toc = std::chrono::system_clock::now();
    if (!remote_duration) {
        printf("Duration: %7.3f ms / (%d images * %d devices)\n\n", std::chrono::duration<double>(toc - tic).count() * 1000 , mini_batch, num_devices);
    }


    
    // C++ Alexnet TEST
    printf("\nC++ Golden Start:\n");
    frac_ptr = 0;
    Blob<DPTYPE> *X_test;
    read_HDF5_4D<DPTYPE>(X_test, "signs_train.h5", "train_set_x",/*normalize*/255,/*Pad*/82, 82);
    Blob<DPTYPE> *Z0_test = new Blob<DPTYPE>(mini_batch);
    Blob<int> *mini_Y_test = new Blob<int>(mini_batch);
    Blob<DPTYPE> *Z1_conv_test = new Blob<DPTYPE>(96, 55, 55);
    Blob_maxPool<DPTYPE> *Z1_pool_test = new Blob_maxPool<DPTYPE>(96, 27, 27, 3, 3, 2, 2, 2, 2);
    Blob<DPTYPE> *Z2_conv_test = new Blob<DPTYPE>(256, 27, 27);
    Blob_maxPool<DPTYPE> *Z2_pool_test = new Blob_maxPool<DPTYPE>(256, 13, 13, 3, 3, 2, 2, 1, 1);
    Blob<DPTYPE> *Z3_conv_test = new Blob<DPTYPE>(384, 13, 13, 1, 1);
    Blob<DPTYPE> *Z4_conv_test = new Blob<DPTYPE>(384, 13, 13, 1, 1);
    Blob<DPTYPE> *Z5_conv_test = new Blob<DPTYPE>(256, 13, 13);
    Blob_maxPool<DPTYPE> *Z5_pool_test = new Blob_maxPool<DPTYPE>(256, 6, 6, 3, 3, 2, 2);
    Blob<DPTYPE> *Z6_test = new Blob<DPTYPE>(4096, 1, 1);
    Blob<DPTYPE> *Z7_test = new Blob<DPTYPE>(4096, 1, 1);
    Blob<DPTYPE> *Z8_test = new Blob<DPTYPE>(6, 1, 1);

    frac_ptr = 0;
    parameter<DPTYPE> *W1_conv_test = new parameter<DPTYPE>(96, 3, 11, 11, 4, 4);
    parameter<DPTYPE> *W2_conv_test = new parameter<DPTYPE>(256, 96, 5, 5, 1, 1);
    parameter<DPTYPE> *W3_conv_test = new parameter<DPTYPE>(384, 256, 3, 3, 1, 1);
    parameter<DPTYPE> *W4_conv_test = new parameter<DPTYPE>(384, 384, 3, 3, 1, 1);
    parameter<DPTYPE> *W5_conv_test = new parameter<DPTYPE>(256, 384, 3, 3, 1, 1);
    parameter<DPTYPE> *W6_test = new parameter<DPTYPE>(4096, 256, 6, 6, 1, 1);
    parameter<DPTYPE> *W7_test = new parameter<DPTYPE>(4096, 4096, 1, 1, 1, 1);
    parameter<DPTYPE> *W8_test = new parameter<DPTYPE>(6, 4096, 1, 1, 1, 1);
    laod_HDF5_4D(*W1_conv_test, "Alexnet.h5", "W1", 2);
    laod_HDF5_4D(*W2_conv_test, "Alexnet.h5", "W2", 2);
    laod_HDF5_4D(*W3_conv_test, "Alexnet.h5", "W3", 2);
    laod_HDF5_4D(*W4_conv_test, "Alexnet.h5", "W4", 2);
    laod_HDF5_4D(*W5_conv_test, "Alexnet.h5", "W5", 2);
    laod_HDF5_4D(*W6_test, "Alexnet.h5", "fc6/weights", 1);
    laod_HDF5_4D(*W6_test, "Alexnet.h5", "fc6/biases", 0);
    laod_HDF5_4D(*W7_test, "Alexnet.h5", "fc7/weights", 1);
    laod_HDF5_4D(*W7_test, "Alexnet.h5", "fc7/biases", 0);
    laod_HDF5_4D(*W8_test, "Alexnet.h5", "fc8/weights", 1);
    laod_HDF5_4D(*W8_test, "Alexnet.h5", "fc8/biases", 0);

    Blob<int> *Y_test;
    read_HDF5_4D<int>(Y_test, "signs_train.h5", "train_set_y", 1);
    mini_Y_test->get_mini_batch(*Y_test);
    Z0_test->get_mini_batch(*X_test);

    tic2 = std::chrono::system_clock::now();
    Convolution(Z1_conv_test, W1_conv_test, Z0_test);
    relu(Z1_conv_test);
    maxPool(Z1_pool_test, Z1_conv_test);
    Convolution(Z2_conv_test, W2_conv_test, Z1_pool_test);
    relu(Z2_conv_test);
    maxPool(Z2_pool_test, Z2_conv_test);
    Convolution(Z3_conv_test, W3_conv_test, Z2_pool_test);
    relu(Z3_conv_test);
    Convolution(Z4_conv_test, W4_conv_test, Z3_conv_test);
    relu(Z4_conv_test);
    Convolution(Z5_conv_test, W5_conv_test, Z4_conv_test);
    relu(Z5_conv_test);
    maxPool(Z5_pool_test, Z5_conv_test);
    Convolution(Z6_test, W6_test, Z5_pool_test);
    relu(Z6_test);
    Convolution(Z7_test, W7_test, Z6_test);
    relu(Z7_test);
    Convolution(Z8_test, W8_test, Z7_test);
    toc2 = std::chrono::system_clock::now();
    if (remote_duration) {
        printf("AlexNet duration: %7.3f ms\n\n", std::chrono::duration<double>(toc2 - tic2).count() * 1000);
    }

    printf("\nPrint Result:\n");
    for (int current_device = 0; current_device < num_devices; current_device++) {
        printf("--- device_%d ---\n", current_device);
        for (int i = 0; i < mini_batch; i++) {
            printf("---------- result %d ---------\n", i);
            for (int j = 0; j < 6; j++) {
#if Fixed_point
                printf("%3d: OpenCL:%6d,   C++:%6d ", j, Z8[current_device]->data[i][j], Z8_test->data[i][j]);
                if (j == mini_Y->data[i][0] >> mini_Y->frac) printf("V");
#else
                printf("%3d: OpenCL:%11.7f,   C++:%11.7f ", j, Z8[current_device]->data[i][j], Z8_test->data[i][j]);
                if (j == mini_Y->data[i][0]) printf("V");
#endif
                printf("\n");
            }
        }
        printf("\n");
    }


	cleanup();

	return 0;
}

void cleanup() {
	if (Conv_kernel) {
		clReleaseKernel(Conv_kernel);
	}
	if (weightRD_kernel) {
		clReleaseKernel(weightRD_kernel);
	}
	if (biasRD_kernel) {
		clReleaseKernel(biasRD_kernel);
	}
	if (dataRD_kernel) {
		clReleaseKernel(dataRD_kernel);
	}
	if (dataWR_kernel) {
		clReleaseKernel(dataWR_kernel);
	}
	if (context) {
		clReleaseContext(context);
	}
	for (int current_device = 0; current_device < num_devices; current_device++) {
		if (program) {
			clReleaseProgram(program[current_device]);
		}
		if (command_queue) {
			clReleaseCommandQueue(command_queue[current_device]);
		}
		if (weightRD_queue) {
			clReleaseCommandQueue(weightRD_queue[current_device]);
		}
		if (biasRD_queue) {
			clReleaseCommandQueue(biasRD_queue[current_device]);
		}
		if (dataRD_queue) {
			clReleaseCommandQueue(dataRD_queue[current_device]);
		}
		if (dataWR_queue) {
			clReleaseCommandQueue(dataWR_queue[current_device]);
		}
		if (Z0_mem) {
			clReleaseMemObject(Z0_mem[current_device]);
		}
		if (W1_conv_mem) {
			clReleaseMemObject(W1_conv_mem[current_device]);
		}
		if (b1_conv_mem) {
			clReleaseMemObject(b1_conv_mem[current_device]);
		}
		if (Z1_pool_mem) {
			clReleaseMemObject(Z1_pool_mem[current_device]);
		}
		if (W2_conv_mem) {
			clReleaseMemObject(W2_conv_mem[current_device]);
		}
		if (b2_conv_mem) {
			clReleaseMemObject(b2_conv_mem[current_device]);
		}
		if (Z2_pool_mem) {
			clReleaseMemObject(Z2_pool_mem[current_device]);
		}
		if (W3_mem) {
			clReleaseMemObject(W3_mem[current_device]);
		}
		if (b3_mem) {
			clReleaseMemObject(b3_mem[current_device]);
		}
		if (Z3_mem) {
			clReleaseMemObject(Z3_mem[current_device]);
		}
		if (W4_mem) {
			clReleaseMemObject(W4_mem[current_device]);
		}
		if (b4_mem) {
			clReleaseMemObject(b4_mem[current_device]);
		}
		if (Z4_mem) {
			clReleaseMemObject(Z4_mem[current_device]);
		}
		if (W5_mem) {
			clReleaseMemObject(W5_mem[current_device]);
		}
		if (b5_mem) {
			clReleaseMemObject(b5_mem[current_device]);
		}
		if (Z5_mem) {
			clReleaseMemObject(Z5_mem[current_device]);
		}
		if (W3_conv_mem) {
			clReleaseMemObject(W3_conv_mem[current_device]);
		}
		if (b3_conv_mem) {
			clReleaseMemObject(b3_conv_mem[current_device]);
		}
		if (Z3_conv_mem) {
			clReleaseMemObject(Z3_conv_mem[current_device]);
		}
		if (W4_conv_mem) {
			clReleaseMemObject(W4_conv_mem[current_device]);
		}
		if (b4_conv_mem) {
			clReleaseMemObject(b4_conv_mem[current_device]);
		}
		if (Z4_conv_mem) {
			clReleaseMemObject(Z4_conv_mem[current_device]);
		}
		if (W5_conv_mem) {
			clReleaseMemObject(W5_conv_mem[current_device]);
		}
		if (b5_conv_mem) {
			clReleaseMemObject(b5_conv_mem[current_device]);
		}
		if (Z5_pool_mem) {
			clReleaseMemObject(Z5_pool_mem[current_device]);
		}
		if (W6_mem) {
			clReleaseMemObject(W6_mem[current_device]);
		}
		if (b6_mem) {
			clReleaseMemObject(b6_mem[current_device]);
		}
		if (Z6_mem) {
			clReleaseMemObject(Z6_mem[current_device]);
		}
		if (W7_mem) {
			clReleaseMemObject(W7_mem[current_device]);
		}
		if (b7_mem) {
			clReleaseMemObject(b7_mem[current_device]);
		}
		if (Z7_mem) {
			clReleaseMemObject(Z7_mem[current_device]);
		}
		if (W8_mem) {
			clReleaseMemObject(W8_mem[current_device]);
		}
		if (b8_mem) {
			clReleaseMemObject(b8_mem[current_device]);
		}
		if (Z8_mem) {
			clReleaseMemObject(Z8_mem[current_device]);
		}
	}
}
