// muladdx4.v

// Generated using ACDS version 17.0 290

`timescale 1 ps / 1 ps
module muladdx4 (
		input   clock,
		input   resetn,
		input   ivalid, 
		input   iready,
		output  ovalid, 
		output  oready,
		input  wire [15:0] dataa_0, // dataa_0.dataa_0
		input  wire [15:0] dataa_1, // dataa_1.dataa_1
		input  wire [15:0] dataa_2, // dataa_2.dataa_2
		input  wire [15:0] dataa_3, // dataa_3.dataa_3
		input  wire [15:0] datab_0, // datab_0.datab_0
		input  wire [15:0] datab_1, // datab_1.datab_1
		input  wire [15:0] datab_2, // datab_2.datab_2
		input  wire [15:0] datab_3, // datab_3.datab_3
		output wire [31:0] result   //  result.result
	);
	assign ovalid = 1'b1;
	assign oready = 1'b1;
	muladdx4_altera_mult_add mult_add_0 (
		.clock0  (clock),   //  clock0.clock0
		.dataa_0 (dataa_0), // dataa_0.dataa_0
		.dataa_1 (dataa_1), // dataa_1.dataa_1
		.dataa_2 (dataa_2), // dataa_2.dataa_2
		.dataa_3 (dataa_3), // dataa_3.dataa_3
		.datab_0 (datab_0), // datab_0.datab_0
		.datab_1 (datab_1), // datab_1.datab_1
		.datab_2 (datab_2), // datab_2.datab_2
		.datab_3 (datab_3), // datab_3.datab_3
		.result  (result)  //  result.result
	);

endmodule

module lpm_mult_ALM (
		input   clock,
		input   resetn,
		input   ivalid, 
		input   iready,
		output  ovalid, 
		output  oready,
   
		input  wire [15:0]  dataa,  //  mult_input.dataa
		input  wire [15:0]  datab,  //            .datab
		output wire [31:0] result  // mult_output.result
	);
	wire [0:0] aclr = {1{~resetn}};
	wire [18:0] result_19b;
	wire [8:0] dataa_9b;
	wire [9:0] datab_10b;
	assign ovalid = 1'b1;
	assign oready = 1'b1;
  
	assign result = {{13{result_19b[18]}}, result_19b};
	assign dataa = {{7{dataa_9b[8]}}, dataa_9b};
	assign datab = {{6{datab_10b[9]}}, datab_10b};

	lpm_mult_lpm_mult_170_xgl63dy lpm_mult_0 (
		.dataa  (dataa_9b),  //  mult_input.dataa
		.datab  (datab_10b),  //            .datab
		.clock  (clock),  //            .clock
		.aclr   (aclr),   //            .aclr
		.result (result_19b)  // mult_output.result
	);

endmodule