#ifndef _IHC_APINT_H_
#define _IHC_APINT_H_
#pragma OPENCL EXTENSION cl_intel_arbitrary_precision_integers : enable
typedef ap_int<9>   int9_t;
typedef ap_int<10>   int10_t;
typedef ap_int<16>   int16_t;
typedef ap_int<18>   int18_t;
typedef ap_int<19>   int19_t;
#endif // _IHC_APINT_H_

#include "muladd.h"
#define N 4
#define KERNEL_PARALLEL 4
#define FILTER_PARALLEL 4

#define POOL_SIZE_W 4
#define POOL_SIZE_H 4
#define IMG_ROW_MAX 31

#define Winograd_Index 3 // filter width unroll factor
#define Winograd_output 2

#define Fixed_point 1
#if Fixed_point
	typedef char DPTYPE;
	typedef short BUFTYPE;
	typedef int MACTYPE;
	#define MASK 0xFF       // used for MAC, FP8
	#define DPTYPE_HI 127
	#define DPTYPE_LO -128
#else
	typedef float DPTYPE;
	typedef float MACTYPE;
	#define DPTYPE_HI INFINITY
	#define DPTYPE_LO -INFINITY
#endif

#if KERNEL_PARALLEL==FILTER_PARALLEL
	#define SAME_PARALLEL 1
#elif KERNEL_PARALLEL>FILTER_PARALLEL
	#define KERNEL_MAIN 1
	#define on_RD 1
#else
	#define FILTER_MAIN 1
#endif

typedef struct{
	DPTYPE kk[KERNEL_PARALLEL];
} channel_parallel;

typedef struct{
	channel_parallel ww[Winograd_output];
} channel_pack;

typedef struct{
	DPTYPE ff[FILTER_PARALLEL];
} filter_parallel;

typedef struct{
	filter_parallel ww[Winograd_Index+Winograd_output-1];
} filter_pack;

typedef struct{
	filter_parallel ww[Winograd_Index];
} filter_pack2;

typedef struct{
	filter_pack2 kk[KERNEL_PARALLEL];
} kernel_pack;

typedef struct{
	BUFTYPE ff[FILTER_PARALLEL];
} filter_trans_w;

typedef struct{
	BUFTYPE ff[FILTER_PARALLEL];
} filter_trans_d;

typedef struct{
	filter_trans_d ww[Winograd_Index+Winograd_output-1];
} data_trans;

typedef struct{
	filter_trans_w ww[Winograd_Index+Winograd_output-1];
} weight_trans;

typedef struct{
	MACTYPE ww[Winograd_output];
} result_pack;

typedef struct{
	MACTYPE kk[KERNEL_PARALLEL];
} buffer;

typedef struct{
	buffer ww[Winograd_output];
} buffer_pack;

typedef struct{
	char send;
	int addr;
	char get_w;
} control;

typedef struct{
	int total_loop;
} information;

#pragma OPENCL EXTENSION cl_intel_channels : enable
channel filter_pack2 weight_trans_ch  __attribute__((depth(0)));
channel weight_trans weight_ch[KERNEL_PARALLEL]  __attribute__((depth(256*6*6/FILTER_PARALLEL)));
channel weight_trans feeder_ch[KERNEL_PARALLEL]  __attribute__((depth(0)));
channel filter_pack data_trans_ch  __attribute__((depth(0)));
channel data_trans data_ch[KERNEL_PARALLEL]  __attribute__((depth(0)));
channel result_pack result_ch[KERNEL_PARALLEL] __attribute__((depth(0)));
channel buffer_pack drain_ch[KERNEL_PARALLEL] __attribute__((depth(0)));
channel channel_parallel bias_ch  __attribute__((depth(0)));
channel channel_pack dataOut_ch  __attribute__((depth(0)));
channel control cont_ch[KERNEL_PARALLEL]  __attribute__((depth(0)));

__kernel
__attribute__((num_compute_units(KERNEL_PARALLEL)))
__attribute__((max_global_work_dim(0))) 
__attribute__((autorun))
void drain(){
	int ID = get_compute_id(0);
	result_pack one_result;
	buffer_pack all_result;
	while(1){
		one_result = read_channel_intel(result_ch[ID]);
		if(ID != 0){
			all_result = read_channel_intel(drain_ch[ID-1]);
		}
		#pragma unroll
		for(int i=0; i<Winograd_output; i++){
			all_result.ww[i].kk[KERNEL_PARALLEL-ID-1] = one_result.ww[i];
		}
		write_channel_intel(drain_ch[ID],all_result);
	}
}

__kernel
__attribute__((num_compute_units(KERNEL_PARALLEL)))
__attribute__((max_global_work_dim(0))) 
__attribute__((autorun))
void feeder(){
	int ID = get_compute_id(0);

	while(1){
		weight_trans temp;
		//#pragma unroll 1
		for( int j=0 ; j < KERNEL_PARALLEL - ID ; j++ ){
			temp = read_channel_intel(feeder_ch[ID]);
			if( j < KERNEL_PARALLEL - ID - 1 ){
				write_channel_intel(feeder_ch[ID+1],temp);
			}
		}
		write_channel_intel(weight_ch[ID], temp);
	}
}

__kernel
__attribute__((num_compute_units(KERNEL_PARALLEL)))
__attribute__((max_global_work_dim(0))) 
__attribute__((autorun))
void PE(){

	int ID = get_compute_id(0);  // each PE denote an output from a kernel of weight
	weight_trans w_local[256*6*6/FILTER_PARALLEL];
	MACTYPE winograd[Winograd_Index+Winograd_output-1];
	MACTYPE result_buffer[Winograd_output];

	// initial shift register
	MACTYPE shift_reg[Winograd_output][N+1];
	#pragma unroll
	for(int k=0;k<Winograd_output; k++){
		#pragma unroll
		for(int j=0;j<N+1;j++){
			shift_reg[k][j]=0;
		}
	}

	while(1){
		data_trans data_in = read_channel_intel(data_ch[ID]); // read data in 1x1xFILTER_PARALLEL (WxHxC)
		control cont = read_channel_intel(cont_ch[ID]);
		
		if(cont.get_w == 1){
			weight_trans get_w_from_channel = read_channel_intel(weight_ch[ID]);
			w_local[cont.addr] = get_w_from_channel;
		}
		weight_trans get_w_from_local = w_local[cont.addr];
		#pragma unroll
		for(int n=0; n<Winograd_Index+Winograd_output-1; n++){
			winograd[n] = 0;
			#pragma unroll
			for(int j=0; j<FILTER_PARALLEL/4; j++){
				winograd[n] += muladdx4(data_in.ww[n].ff[j*4],get_w_from_local.ww[n].ff[j*4],data_in.ww[n].ff[j*4+1],get_w_from_local.ww[n].ff[j*4+1],data_in.ww[n].ff[j*4+2],get_w_from_local.ww[n].ff[j*4+2],data_in.ww[n].ff[j*4+3],get_w_from_local.ww[n].ff[j*4+3]);
			}
		}

		result_buffer[0] = (winograd[0] + winograd[1] + winograd[2]);
		result_buffer[1] = (winograd[1] - winograd[2] - winograd[3]);

		// shift partial_sum
		shift_reg[0][N] = shift_reg[0][0] + result_buffer[0];
		shift_reg[1][N] = shift_reg[1][0] + result_buffer[1];

		#pragma unroll
		for(int i=0;i<N;i++){
			shift_reg[0][i] = shift_reg[0][i+1];
			shift_reg[1][i] = shift_reg[1][i+1];
		}

		if(cont.send == 1){
			#pragma unroll
			for(int i=0; i<Winograd_output; i++){
				#pragma unroll
				for(int j=1;j<N;j++){
					shift_reg[i][0] += shift_reg[i][j];
				}
			}
			result_pack result;
			#pragma unroll
			for(int i=0; i<Winograd_output; i++){
				result.ww[i] = shift_reg[i][0];
			}

			write_channel_intel(result_ch[ID], result);
			#pragma unroll
			for(int i=0; i<Winograd_output; i++){
				#pragma unroll
				for( int j = 0 ; j < N ; j++ ){
					shift_reg[i][j] = 0;
				}
			}
		}

		if(ID < KERNEL_PARALLEL-1){
			write_channel_intel(data_ch[ID+1], data_in);
			write_channel_intel(cont_ch[ID+1], cont);
		}
	}
}


__kernel 
__attribute__((task)) 
__attribute__((max_global_work_dim(0))) 
void weightRD(__global const filter_pack2 * restrict W, 
	int W_kernel_size, int Z_h_loop, int Z_k_loop,
	int Weight_size) 
{
	for ( int c = 0 ; c < Weight_size ; c++ ) {
		filter_pack2 temp = W[c];
		write_channel_intel(weight_trans_ch,temp);
	}
}

__kernel
__attribute__((max_global_work_dim(0))) 
__attribute__((autorun))
void weight_transform(){
	
	filter_pack2 w_in;
	weight_trans w_trans;
	while(1){
		w_in = read_channel_intel(weight_trans_ch);
		#pragma unroll
		for(int j=0; j<FILTER_PARALLEL; j++){
			w_trans.ww[0].ff[j] = (BUFTYPE)(w_in.ww[0].ff[j]<<1);
			w_trans.ww[1].ff[j] = (BUFTYPE)(w_in.ww[0].ff[j] + w_in.ww[1].ff[j] + w_in.ww[2].ff[j]);
			w_trans.ww[2].ff[j] = (BUFTYPE)(w_in.ww[0].ff[j] - w_in.ww[1].ff[j] + w_in.ww[2].ff[j]);
			w_trans.ww[3].ff[j] = (BUFTYPE)(w_in.ww[2].ff[j]<<1);
		}
		write_channel_intel(feeder_ch[0], w_trans);
	}
}

__kernel 
__attribute__((task)) 
__attribute__((max_global_work_dim(0))) 
void biasRD(__global const channel_parallel * restrict b, 
	int Kernel) 
{
	for(int n=0; n<Kernel; n++){
		channel_parallel temp = b[n];
		write_channel_intel(bias_ch,temp);
	}
}

__kernel 
__attribute__((task)) 
__attribute__((max_global_work_dim(0))) 
void dataRD(
	__global const filter_parallel * restrict Z_prev, 
	int filter_w, int prev_Width_pad,
	int Z_prev_img_size, int Z_prev_size, int W_filter_size, 
	int W_kernel_size, int Z_w_loop, 
	int Z_h_loop, int Z_batch_loop, 
	int Z_k_loop, int prev_w_step, int prev_h_step) 
{

#if KERNEL_MAIN && on_RD
	int Parallel_count = (KERNEL_PARALLEL / FILTER_PARALLEL);
	int parallel_step = 0;
	int step = KERNEL_PARALLEL / FILTER_PARALLEL;
#else
	int step = 1;
#endif

	for (int w=0,fW=0,fH=0,c=0,prev_batch_start=0,prev_h_start=0,prev_w_start=0,prev_slice_c_start=0,prev_slice_h_start=0,batch_loop=0,w_loop=0,h_loop=0,k_loop=0 ; k_loop < Z_k_loop;) {
		
		if(fW==filter_w){
			fW = 0;
			w=0;
			prev_slice_h_start += prev_Width_pad;
		}
		if(fH==W_filter_size){
			fH=0;
			prev_slice_h_start = 0;
#if KERNEL_MAIN && on_RD
			parallel_step++;
#else
			prev_slice_c_start +=  Z_prev_img_size;
#endif
		}
#if KERNEL_MAIN && on_RD
		if(parallel_step == Parallel_count){
			parallel_step = 0;
			prev_slice_c_start +=  Z_prev_img_size;
		}
#endif
		if(c==W_kernel_size){
			c=0;
			prev_w_start += prev_w_step;
			prev_slice_c_start = prev_w_start;	
#if KERNEL_MAIN && on_RD
			parallel_step = 0;
#endif
		}
		if(w_loop==Z_w_loop){
			w_loop = 0;
			prev_h_start += prev_h_step;
			prev_w_start = prev_h_start;
			prev_slice_c_start = prev_w_start;
		}
		if(h_loop==Z_h_loop){
			h_loop = 0;
			prev_batch_start += Z_prev_size;
			prev_h_start = prev_batch_start;
			prev_w_start = prev_h_start;
			prev_slice_c_start =  prev_w_start;
#if KERNEL_MAIN && on_RD
			parallel_step = 0;
#endif
		}
		if(batch_loop == Z_batch_loop){
			batch_loop = 0;
			prev_batch_start = 0;
			prev_h_start = 0;
			prev_w_start = 0;
			prev_slice_c_start = 0;
		}
		control cont;
		if(c+step==W_kernel_size){
			cont.send = 1;
		}else{
			cont.send = 0;
		}
		if(batch_loop < W_kernel_size){
			cont.get_w = 1;
		}else{
			cont.get_w = 0;
		}
		cont.addr = c;

#if KERNEL_MAIN && on_RD
		filter_pack data;
		#pragma unroll
        for(int i=0; i<Winograd_Index+Winograd_output-1; i++){
            filter_parallel Z_data_cache = Z_prev[prev_slice_c_start + prev_slice_h_start + w + i + parallel_step];
            data.ww[i] = Z_data_cache;
        }
#else
		filter_pack data = *((filter_pack *) (&Z_prev[prev_slice_c_start + prev_slice_h_start + w]));
#endif
		w+=Winograd_Index,fW+=step,fH+=step,c+=step,batch_loop+=step,w_loop+=step,h_loop+=step,k_loop+=step;
		write_channel_intel(data_trans_ch,data);
		write_channel_intel(cont_ch[0],cont);
	}
}

__kernel
__attribute__((max_global_work_dim(0))) 
__attribute__((autorun))
void data_transform(){
	
	filter_pack data_in;
	data_trans data_trans;
	while(1){
		data_in = read_channel_intel(data_trans_ch);
		#pragma unroll
		for(int j=0; j<FILTER_PARALLEL; j++){
			data_trans.ww[0].ff[j] = (BUFTYPE)(data_in.ww[0].ff[j] - data_in.ww[2].ff[j]);
			data_trans.ww[1].ff[j] = (BUFTYPE)(data_in.ww[1].ff[j] + data_in.ww[2].ff[j]);
			data_trans.ww[2].ff[j] = (BUFTYPE)(data_in.ww[2].ff[j] - data_in.ww[1].ff[j]);
			data_trans.ww[3].ff[j] = (BUFTYPE)(data_in.ww[1].ff[j] - data_in.ww[3].ff[j]);
		}
		write_channel_intel(data_ch[0], data_trans);
	}
}

__kernel 
__attribute__((task)) 
__attribute__((max_global_work_dim(0))) 
void dataWR(
#if KERNEL_MAIN && !on_RD
	__global filter_parallel * restrict Z,
#else
	__global channel_parallel * restrict Z,
#endif
	int Z_w_loop, int Z_h_loop, int Z_batch_loop, int Z_k_loop,
	int next_image_group, int next_image, int next_size,
	int next_pad_h, int next_pad_w, int next_Width_pad, int next_w_step,
	int stride_h, int stride_w, int pool_h, int pool_w,
	int stride_h_init, int stride_w_init,
	int output_num
	) 
{

	channel_parallel pool_row[IMG_ROW_MAX][POOL_SIZE_H];
	DPTYPE pool_reg[KERNEL_PARALLEL][POOL_SIZE_W+1];
	channel_pack temp;

	#pragma unroll 1
	for(int i=0;i<IMG_ROW_MAX;i++){
		#pragma unroll
		for(int j=0;j<POOL_SIZE_H;j++){
			#pragma unroll
			for(int l=0;l<KERNEL_PARALLEL;l++){
				pool_row[i][j].kk[l] = DPTYPE_LO;
			}
		}
	}

	for (int out = 0, k_loop = 0, h = next_pad_h, w = next_pad_w, k = 0, batch = 0, batch_loop = 0, stride_w_current = 0, stride_h_current = 0, stride_w_loop = stride_w_init, stride_h_loop = stride_h_init, h_loop = 0, w_loop = 0, batch_start = 0, k_start = 0, h_start = next_w_step; k_loop < Z_k_loop;) {

		if (w_loop == Z_w_loop) {
			stride_w_loop = stride_w_init;
			w = next_pad_w;
			w_loop = 0;
			stride_h_loop++;
			if (stride_h_loop == stride_h) {
				stride_h_loop = 0;
				h++;
				h_start += next_Width_pad;
			}
			stride_h_current++;
			if(stride_h_current == pool_h){
				stride_h_current = 0;
			}
		}
		if (h_loop == Z_h_loop) {
			stride_h_loop = stride_h_init;
			stride_h_current = 0;
			h_loop = 0;
			h = next_pad_h;
			batch_start += next_size;
			batch++;
			h_start = next_w_step;
		}
		if ( batch_loop == Z_batch_loop){
			batch_loop = 0;
			batch = 0;
#if KERNEL_MAIN && !on_RD
			k+=KERNEL_PARALLEL/FILTER_PARALLEL;
#else
			k++;
#endif
			k_start += next_image_group;
			batch_start = k_start;
			h_start = next_w_step;
		}
		if(out == output_num || w_loop == 0){ //sometimes Width can't devided by output_num, when w_loop=0, just abandon rest of output and get new one.
			out = 0;
		}
		if(out == 0){
			temp = read_channel_intel(dataOut_ch);
		}

		#pragma unroll
		for(int l=0;l<KERNEL_PARALLEL;l++){
			pool_reg[l][0] = temp.ww[out].kk[l];
		}
		
		#pragma unroll
		for(int l=0;l<KERNEL_PARALLEL;l++){
			#pragma unroll
			for (int i = POOL_SIZE_W; i>0; i--) {
				pool_reg[l][i] = pool_reg[l][i - 1];
			}
		}

		#pragma unroll
		for(int l=0;l<KERNEL_PARALLEL;l++){
			#pragma unroll
			for(int i = 0; i < POOL_SIZE_W; i++){
				pool_reg[l][i+1] = (pool_reg[l][i] > pool_reg[l][i+1]) ? pool_reg[l][i] : pool_reg[l][i+1];
			}
		}

		out++;
		stride_w_loop++;
		if (stride_w_loop == stride_w) {
			stride_w_loop = 0;

			#pragma unroll
			for(int l=0;l<KERNEL_PARALLEL;l++){
				pool_row[w][stride_h_current].kk[l] = pool_reg[l][pool_w];
			}
			if (stride_h_loop == stride_h-1) {
				channel_parallel temp2;
				#pragma unroll
				for(int l=0;l<KERNEL_PARALLEL;l++){
					temp2.kk[l] = DPTYPE_LO;
				}
				#pragma unroll
				for(int i=0;i<POOL_SIZE_H;i++){
					#pragma unroll
					for(int l=0;l<KERNEL_PARALLEL;l++){
						temp2.kk[l] = (pool_row[w][i].kk[l] > temp2.kk[l]) ? pool_row[w][i].kk[l] : temp2.kk[l];
					}
				}
#if KERNEL_MAIN && !on_RD
				int batch_offset = 0;
				#pragma unroll
				for(int i=0; i<KERNEL_PARALLEL/FILTER_PARALLEL; i++){
					#pragma unroll
					for(int l=0;l<FILTER_PARALLEL;l++){
						Z[batch_start + h_start + w + batch_offset].ff[l] = temp2.kk[ i * FILTER_PARALLEL + l ];
					}
					batch_offset += next_image;
				}
#else
				Z[batch_start + h_start + w] = temp2;
				int idx = 0;
#endif
			}
			w++;
		}
		w_loop++, h_loop++, k_loop++, batch_loop++;
	}
}

__kernel 
__attribute__((task)) 
__attribute__((max_global_work_dim(0))) 
void conv_core( 
	int W_kernel_size, int Z_h_loop, int Z_k_loop,
	int Z_h_loop_sys, int Z_k_loop_sys, 
	int relu
#if Fixed_point
	,int frac
#endif
	) 
{
	
	channel_parallel b_cache;
	channel_pack Z_data_cache;
	buffer_pack result_buffer;

	for (int k_loop = 0,h_loop = 0; k_loop < Z_k_loop_sys; ) {

		if(h_loop==Z_h_loop_sys || h_loop==0){
			h_loop = 0;
			b_cache = read_channel_intel(bias_ch);
		}

		h_loop++,k_loop++;  // virtual loop counter

		result_buffer = read_channel_intel(drain_ch[KERNEL_PARALLEL-1]);

		#pragma unroll
		for(int i=0; i<KERNEL_PARALLEL; i++){
#if Fixed_point
			result_buffer.ww[0].kk[i] += (b_cache.kk[i] << (frac+1));
			result_buffer.ww[1].kk[i] += (b_cache.kk[i] << (frac+1));
#else
			result_buffer.ww[0].kk[i] += b_cache.kk[i];
			result_buffer.ww[1].kk[i] += b_cache.kk[i];
#endif
		}
#if Fixed_point
		MACTYPE temp_cache[Winograd_output];
		#pragma unroll
		for(int k=0;k<Winograd_output;k++){
			#pragma unroll
			for(int i=0;i<KERNEL_PARALLEL;i++){
				temp_cache[k] = ((result_buffer.ww[k].kk[i] >> frac) + 0x1) >> 1;
				if(temp_cache[k] > DPTYPE_HI ){
					Z_data_cache.ww[k].kk[i] = DPTYPE_HI;
				}else if(temp_cache[k] < DPTYPE_LO){
					Z_data_cache.ww[k].kk[i] = DPTYPE_LO;
				}else{
					Z_data_cache.ww[k].kk[i] = temp_cache[k];
				}
			}
		}
#else
		#pragma unroll
		for(int k=0;k<Winograd_output;k++){
			#pragma unroll
			for(int i=0;i<KERNEL_PARALLEL;i++){
				Z_data_cache.ww[k].kk[i] = result_buffer.ww[k].kk[i];
			}
		}
#endif
		if(relu){
			#pragma unroll
			for(int k=0;k<Winograd_output;k++){
				#pragma unroll
				for(int i=0;i<KERNEL_PARALLEL;i++){
					Z_data_cache.ww[k].kk[i] = (Z_data_cache.ww[k].kk[i] > 0) ? Z_data_cache.ww[k].kk[i] : 0 ;
				}
			}
		}
		write_channel_intel(dataOut_ch, Z_data_cache);
	}
}
