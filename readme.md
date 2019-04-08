This project was created in the writing of an academic thesis, so the organizational structure and documentation may be lacking. Follow these directions to get started:

1. data.py
  * Set filepath to your data csv -- this csv should have the same form as in the first figure of section 2.4.1 in the accompanying 'paper.pdf'
  * Set savepath to where you want the output (e.g. '~/csv') -- the output will be in csv form, and will look like the second figure in section 2.4.1
  * Run the main(bool) function, which takes a boolean argument to determine whether you want to save a csv (I recommend setting it to False while testing)
  * **IMPORTANT:** change 'RET' in line 33 to 'PRC' and repeat the above procedure to generate a matrix of returns
2. techs.py
  * Set projectpath to your root and filepath to the prices csv generated in step 1
  * Run the main(a, b, c) function, which takes integer window lengths a, b, and c as arguments (these windows will be applied to the Moving Average and Rolling Min/Max indicators)
3. signals.py
  * Set projectpath to your root, csvpath to where you saved the output of step 1 (e.g. '~/csv'), and filepath to the actual csv output from step 1 (e.g.'~/csv/data_out.csv')
  * Run the main(a, b, c, j) function, which takes the same three window lengths a, b, and c from step 2, and an additional float parameter j which determines the threshold at which an asset price is considered "close" to a technical indicator (1% was used in my study)
4. stats.py
  * Set projectpath and csvpath to the same as in the previous step; set filepath equal the matrix of **returns** (not prices) generated in step 1
  * Run the main() function, which generates pairs distributions of returns for each security across each technical indicator, and saves it as a pickle file ('container')
6. output.py
  * Set projectpath to the same as in the previous step; set containerpath to the pickle file that was generated in step 4
  * Run the stats_tests() function to perform the included HOV tests on each security (Brown-Forsythe, Bartlett, Levene, and Fligner-Killeen)
  * Run the adjust_p() function to adjust p-values for multiple sampling
  * The remaining functions (starting on line 164) generate the visualizations seen in the accompanying 'paper.pdf'
7. tables.py
  * Set projectpath and containerpath to the same as in the previous step; set returns_out to the matrix of returns generated in step 1
  * All functions from line 22 through line 185 generate the tables seen in the accompanying 'paper.pdf'
