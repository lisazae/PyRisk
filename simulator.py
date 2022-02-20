"""
Simulacion de las variables que posiblemente afecten el comportamiento de 
un problema particular

"""

from ast import Param
from math import ceil
from optparse import Values
from pydoc import describe
from tkinter import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

from pip import main
from pyparsing import col
import scipy.stats as st
import seaborn as sns


class Simulator:
    """
    Responsable de albergar todos los métodos que permiten obtener
    los resultados de la simulación y con ella ejecutar los resumenes y gráficos 
    para la toma de decisiones sobre un problema particular.
    
    :param nsim:
        Número de simulaciones.

    :param niter:
        Número de iteraciones.

    :param fn_generadora:
        Función que genera el usuario que realiza el cálculo de los valores que
        se desean simular, este resultado debe arrojar un diccionario, 
        donde la clave sea el nombre de la variable a simular y el valor los resultados. 

    :param variables_entrada:
        Función que retorna un diccionario con la cantidad de variables importantes
        para describir el problema, la clave del diccionario es el nombre de la 
        variable y el valor la distribución de probabilidad que describe el comportamiento de 
        esta con sus respectivos valores.

    :param alpha:
        Confianza con la cual se va a realizar el intervalo de confianza para la media.
    """
    def __init__(self,nsim,niter,fn_generadora,variables,alpha):
        self.nsim=nsim
        self.niter=niter
        self.fn_generadora=fn_generadora
        self.variables=variables
        self.alpha=alpha
        self.result_simulation=None
    
    def get_simulations(self):
        """
        Con esta meétodo es posible obtener los resultados netos de la simulación, es decir, 
        sin ningún cálculo o transformación. 
        """
        df=pd.DataFrame(self.result_simulation)
        return df

    def run_simulation(self):
        """
        Este método es responsable de ejecutar la función f_generadora y recolectar los resultados. La función va a realizar n_sim
        simulaciones y cada una de ellas de n_iter iteraciones. La función retorna los resultados en el siguiente formato:
        una lista de diccionarios donde cada diccionario de n_sim posiciones. Cada elemento de la lista un diccionario que recoge la n_iter para
        cada una de las variablexs, es decir, si hay 2 simulaciones cada una de 5 iteraciones y hay 1 variable para analizar, la
        estructura que retorna es la siguiente:   
        
        [{'simulacion':0, 'iteracion': 0, 'VPN': 48400.10987514735},
        {'simulacion': 0, 'iteracion': 1, 'VPN': -162646.12417961366}, 
        {'simulacion': 0, 'iteracion': 2, 'VPN': -284233.65396675933}, 
        {'simulacion': 0, 'iteracion': 3, 'VPN': 1831327.3023823607}, 
        {'simulacion': 0, 'iteracion': 4, 'VPN': -549780.3871181059}, 
        {'simulacion': 1, 'iteracion': 0, 'VPN': -106489.52598146687}, 
        {'simulacion': 1, 'iteracion': 1, 'VPN': 296278.747531264}, 
        {'simulacion': 1, 'iteracion': 2, 'VPN': -360814.57651430386}, 
        {'simulacion': 1, 'iteracion': 3, 'VPN': -40666.52610107139}, 
        {'simulacion': 1, 'iteracion': 4, 'VPN': 354583.8758478068}]
        """
        resultados_simulaciones=[]

        for i_sim in range(self.nsim):
            for i_iter in range(self.niter):
                corrida=dict(simulacion=i_sim,iteracion=i_iter)
                resultados_f=self.fn_generadora(**self.variables())
                resultados_simulaciones.append({**corrida,**resultados_f})
            
        self.result_simulation = resultados_simulaciones


    def summary_table(self):
        """
        Este método define la tabla resumen de las diferentes simulaciones e iteraciones
        que se hacen en el simulador. La estructura resultado es la siguiente: 

           simulacion   variable      Mínimo  ...            Q2            Q3           IQR
                    0      TIR -1.562043e+06  ...  1.034710e+06  2.010893e+06  1.765772e+06
                    0      VPN -3.237218e+13  ...  5.550969e+13  9.769850e+13  7.625594e+13
                    1      TIR -1.574251e+06  ...  8.908823e+05  2.035339e+06  1.850693e+06
                    1      VPN -3.515448e+13  ...  4.918859e+13  1.007781e+14  8.011238e+13
                    2      TIR -1.464660e+06  ...  1.061807e+06  2.137692e+06  1.875725e+06
                    2      VPN -3.202647e+13  ...  5.433118e+13  1.036741e+14  8.129621e+13
        """
        
        Base=pd.DataFrame(self.result_simulation)
        Base1 = pd.melt(Base, id_vars=["simulacion"], value_vars=Base.columns[2:])
        
        summary_t1= pd.DataFrame(Base1.groupby(["simulacion","variable"]).agg({"value":["min","max","mean","std","var",
                                lambda x: pd.Series.skew(x),lambda x: pd.Series.kurtosis(x),
                                lambda x: pd.Series.mode(x)[0],lambda x: x.quantile(0.5),
                                lambda x:st.t.interval(self.alpha,len(x) - 1, loc=np.mean(x), scale=st.sem(x)),
                                lambda x: x.quantile(0.25),lambda x: x.quantile(0.5),lambda x: x.quantile(0.75),
                                lambda x: x.quantile(0.75)-x.quantile(0.25)]})).reset_index()
        
        summary_t1.columns=["simulacion","variable","Mínimo",'Máximo','Media','Desviación est','Varianza',"Asimetría",
        'Curtosis','Moda','Mediana','IC','Q1','Q2','Q3','IQR']
       
        df=pd.DataFrame(summary_t1)

        return df    

    def histogram(self,**histplot_kwargs):
        """
        Este meétodo permite realizar los histogramas de los resultados de las simulaciones
        con el fin de observar el comportamiento de estas. 
        """
        
        Base=pd.DataFrame(self.result_simulation)
        Base1 = pd.melt(Base, id_vars=["simulacion"], value_vars=Base.columns[2:])
        Base1=Base1.sort_values('variable')
        

        g=sns.FacetGrid(Base1,col="variable", margin_titles=True,aspect=4,sharex=False)
        plt.gcf().set_size_inches(11.7, 8.27)
        g.map(sns.histplot, "value",**histplot_kwargs)
        g.set_axis_labels("Values","")

        neg=(Base1[Base1.value <0])
        valores=pd.DataFrame(neg.groupby(['variable']).agg({'value':['count']}))/self.niter
        valores=valores.sort_values('variable')
        
        lines_position=valores.values.tolist()

        for ax, pos in zip(g.axes.flat, lines_position):
            ax.axvline(x=pos, color='r', linestyle='--',linewidth=3)
        plt.show()
    
    def correlation(self,method):
        Base = pd.DataFrame(self.result_simulation)
        Base = Base.iloc[:, 2:]
        corr_df = Base.corr(method=method)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        sns.heatmap(corr_df, ax= ax,annot=True, square=True,cbar= False,annot_kws = {"size": 8},
                cmap= sns.diverging_palette(20, 220, n=200),)

        ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right',
        )

        ax.tick_params(labelsize=10)

        plt.title("Correlation matrix")
        plt.show()
        

    def dispersion(self):
        Base=pd.DataFrame(self.result_simulation)
        Base=Base.iloc[:,2:]
        sns.pairplot(Base, height=1.5, corner=True,diag_kind = 'kde')
        plt.title("Dispersion matrix")
        plt.show()
    
    def plot_matrix(self, plot_type='pairplot', columns=False,method='pearson'):
        """
        Método que permite graficar la correlación y dispersión entre variables.

        :param plot_type:
            Por defecto gráfica la matriz de dispersión, pero si se desea el gráfico
            de la correlación se debe cambiar por 'corr_plot'.

        :param method:
            Por defecto el método por el cual se calcula la correlación entre
            variables es la de perason, pero se puede hacer uso de los demás métodos
            'kendall' o 'spearman'.
        """
        sns.set()
        plt.rc("figure", figsize=(16, 8.65))
        df=pd.DataFrame(self.result_simulation)
        df = df.iloc[:, 2:]
        plotting_df = (df[columns] if columns else df)
        
        if plot_type == 'pairplot':
            sns.pairplot(plotting_df, height=5, corner=False,diag_kind = 'kde',kind='scatter')
            plt.suptitle('Dispersion matrix',size=15)
            
        elif plot_type == 'corr_plot':
            corr_df = plotting_df.corr(method=method)
            ax=sns.heatmap(corr_df, annot=True, square=True, cbar=False, annot_kws={"size": 15},
                    cmap=sns.diverging_palette(20, 220, n=200), )

            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=45,
                horizontalalignment='right',
            )
            plt.title("Correlation matrix",size=15)
            ax.tick_params(labelsize=10)
        plt.show()
