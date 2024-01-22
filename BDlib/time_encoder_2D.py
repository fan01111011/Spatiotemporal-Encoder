import numpy as np

def four_corner(base, s_row, s_col, gap_on_row, gap_on_col):

   sliced_hdv = base[s_row: s_row + gap_on_row + 2, s_col: s_col + gap_on_col + 2]
   top_left_kernel = sliced_hdv[0][0]
   top_right_kernel = sliced_hdv[0][-1]
   bot_left_kernel = sliced_hdv[-1][0]
   bot_right_kernel = sliced_hdv[-1][-1]

   for row in range(0 , sliced_hdv.shape[0]): # iter the row
      for col in range(0 , sliced_hdv.shape[1]): # iter the col
         
         if row % (gap_on_row + 1) != 0 or col % (gap_on_col + 1) != 0:

               copy_distance_to_top = int((1 - row / (sliced_hdv.shape[0] - 1)) * (sliced_hdv.shape[2]))
               copy_distance_to_left = int((1 - col / (sliced_hdv.shape[1] - 1)) * (sliced_hdv.shape[3] )) 
               # print(copy_distance_to_top, copy_distance_to_left)
               # copy_distance_to_bot = int(row / (sliced_hdv.shape[0] - 1) * (sliced_hdv.shape[0] - 1))
               # copy_distance_to_right = int(col / (sliced_hdv.shape[1] - 1) * (sliced_hdv.shape[1] - 1))

               # print(copy_distance_to_top, copy_distance_to_bot, copy_distance_to_left, copy_distance_to_right )
               
               # top left parts
               sliced_hdv[row][col][: copy_distance_to_top, : copy_distance_to_left] = top_left_kernel[: copy_distance_to_top, : copy_distance_to_left]

               # top right parts
               sliced_hdv[row][col][: copy_distance_to_top, copy_distance_to_left: ] = top_right_kernel[: copy_distance_to_top, copy_distance_to_left:]
               
               # bot left parts
               sliced_hdv[row][col][copy_distance_to_top: , : copy_distance_to_left] = bot_left_kernel[copy_distance_to_top:, : copy_distance_to_left]

               # bot right parts
               sliced_hdv[row][col][copy_distance_to_top: , copy_distance_to_left: ] = bot_right_kernel[copy_distance_to_top: , copy_distance_to_left: ]
         # print(row, col )
         # print(sliced_hdv[row][col])

   return sliced_hdv

    



def gen_base_2D( num_row, num_col, HD_dim_row, HD_dim_col, gap_on_row, gap_on_col): 

   base_2D = np.zeros((num_row ,num_col, HD_dim_row, HD_dim_col)).astype(np.int32) # generate a 4D vector (row * col) * (sqrt of HD * sqrt of HD) 

   # suppose we have a (16 * 16) * (5 * 5 HD dim)
   # gap between kernel is 3


   for row in range(0 , num_row): # iter the row
      for col in range(0 , num_col): # iter the col

         # if row % (gap_on_row + 1) == 0 and col % (gap_on_col + 1) == 0: # if this is a kernel position
         #       for i in range(0 , HD_dim_row):
         #          for j in range(0 , HD_dim_col):
         #             mu, sigma = 0, 1 # mean and standard deviation
         #             s = np.random.normal(mu, sigma, 1)
         #             base_2D[row][col][i][j] = s



         if row % (gap_on_row + 1) == 0 and col % (gap_on_col + 1) == 0: # if this is a kernel position
               for i in range(0 , HD_dim_row):
                  for j in range(0 , HD_dim_col):
                     if np.random.randint(0 , 10) > 4:
                           base_2D[row][col][i][j] = 1
                     else:
                           base_2D[row][col][i][j] = -1

         # if row == 0 and col == 0:
         #     base_2D[row][col] = 0
         # if row == 0 and col == (gap_on_col + 1):
         #     base_2D[row][col] = 1
         # if row == (gap_on_row + 1) and col == 0:
         #     base_2D[row][col] = 2
         # if row == (gap_on_row + 1) and col == (gap_on_col + 1):
         #     base_2D[row][col] = 3


               # print(base_2D[row][col])
   
   for row in range(0 , num_row - 1):
      for col in range(0, num_col - 1):
         if row % (gap_on_row + 1) == 0 and col % (gap_on_col + 1) == 0:
               base_2D[row : row + gap_on_row + 2 , col : col + gap_on_col + 2] = four_corner(base_2D, row, col, gap_on_row, gap_on_col)
               # print(base_2D[row : row + gap_on_row + 1, col : col + gap_on_col + 1].shape)
               # print(four_corner(base_2D, row, col, gap_on_row, gap_on_col).shape)
   return base_2D




class time_encoder_2D:

   def __init__(self, input_data_dim, output_data_dim,input_amount_of_level_hdv, level_vec_randomness, input_time_length, gap):

      self.data_in_dim = input_data_dim # col or row
      self.data_out_dim = output_data_dim
      self.amount_of_level_hdv = input_amount_of_level_hdv
      self.level_vec_randomness = level_vec_randomness
      self.time_length = input_time_length

      ############################################## Spatiotemporal HDV  ##############################################

      # self.ID_hdv = gen_base_2D( input_data_dim, input_data_dim, 100, 100, gap, gap).reshape(( input_data_dim*input_data_dim , 100, 100))

      # self.ID_hdv = self.ID_hdv.reshape(input_data_dim*input_data_dim,output_data_dim)
 
      #####################################################################################################################


      ############################################## Random HDV  ##############################################

      self.ID_hdv = self.hdv_random()

      #########################################################################################################


      self.level_hdv = self.hdv_level(self.amount_of_level_hdv, level_vec_randomness)


   def hdv_random(self):
      result_arr = np.random.rand( self.data_in_dim ** 2 , self.data_out_dim)
      result_arr = np.where(result_arr > 0.5, 1, -1)
      return result_arr


   def hdv_level(self, amount_of_level_hdvs, randomness):

      result_arr = np.zeros((amount_of_level_hdvs , self.data_out_dim))
      
      result_arr[0] = np.random.rand( self.data_out_dim)
      
      result_arr[0] = np.where(result_arr[0] > 0.5, 1, -1)

      flip_tabel = np.zeros((self.data_out_dim))

      flip_amount = np.floor(self.data_out_dim / (1 / randomness ) / (amount_of_level_hdvs - 1))

      for i in range(1 , amount_of_level_hdvs):

         result_arr[i] = result_arr[i - 1]

         flipped_bits = 0

         while flipped_bits < flip_amount:

            filp_index = np.random.randint(self.data_out_dim)

            if flip_tabel[filp_index] == 0:

               result_arr[i][filp_index] = (0 - result_arr[i][filp_index])

               flipped_bits = flipped_bits + 1

               flip_tabel[filp_index] = 1

      return result_arr
   
   def encode(self, input_data):
      
      result_arr = np.zeros(self.data_out_dim)

      for t in range(0 , self.time_length):

         framed_data = input_data[t]

         for dim_in in range(0 , self.data_in_dim):
            
            # level_use_this_round = (framed_data[dim_in]) / (1.0) * self.amount_of_level_hdv

            level_use_this_round = (framed_data[dim_in] + 1) / (2) * self.amount_of_level_hdv

            if level_use_this_round >= self.amount_of_level_hdv:

               level_use_this_round = level_use_this_round - 1

            temp_arr = np.multiply(self.level_hdv[int(level_use_this_round)], self.ID_hdv[t * self.data_in_dim + dim_in ])

            result_arr = np.add(result_arr, temp_arr)
      
      # result_arr = np.cos(result_arr)
      
      return result_arr