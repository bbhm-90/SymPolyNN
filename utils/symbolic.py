from sympy import latex
def save_latex_table(expr_list, fild_add:str):
    """
        expr_list: list of tuples. first elem of tupe is eq complx (int)
                    sencodn elem is sympy expression
    """
    assert fild_add.endswith(".tex"), fild_add
    with open(fild_add, 'w') as latex_file:
        latex_code = r'''\begin{center}
        \begin{tabular}{|c|c|c|}
        \hline
        Complexity & Loss & Expression \\
        \hline
        '''
        for i in range(len(expr_list)):
            latex_code += str(expr_list[i][0]) + ' & '
            loss_str = '{:.3e}'.format(expr_list[i][1])
            latex_code += loss_str + ' & '
            latex_code += r'$\begin{aligned}' + latex(expr_list[i][2]) + r'\end{aligned}$'
            latex_code += r'\\ \hline'
        latex_code += r'''\end{tabular}
        \end{center}
        '''
        # Write the LaTeX code to the output file
        latex_file.write(latex_code)