from fastapi import FastAPI
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn import tree
from datetime import date
from datetime import datetime
from sklearn.model_selection import train_test_split
from typing import Any, Dict, AnyStr, List, Union
from pydantic import BaseModel
from sklearn.tree import DecisionTreeClassifier
import numpy as np

app = FastAPI()
df = pd.read_excel("2. Datos para entrenar.xlsx",engine='openpyxl')
filtro=[]
for i,j in enumerate(df['fecha_nacimiento']):
    filter=type(df['fecha_nacimiento'][0])
    if type(j)==filter:
        filtro.append(True)
    else:
        filtro.append(False) 
df=df[filtro]
edad=[]
for k,h in enumerate(df['fecha_nacimiento']):
    edad.append(2022-h.year)
df['Edad']=edad

salidas1= df[['consulta_id','cups_solictud','descripcion','cantidad','prescripcion','tipo','codigo_medicamento','descripcion_medicamento','cantidad_medicamento','prescripcion_medicamento']]
entradas= df.drop(['diagnostico_secundario','diagnostico_terciario','cuarto_diagnostico','presentacion','fecha_nacimiento','cita_id','id_solicitud_enviada','medicamento_enviado',"TEST_HERRERA_Y_HURTADO",'cups_solictud','descripcion','cantidad','prescripcion','tipo','codigo_medicamento','descripcion_medicamento','cantidad_medicamento','prescripcion_medicamento'], axis=1)
salidas1=salidas1.fillna(0)
entradas=entradas.fillna(0)
cups_solictud=[]
descripcion=[]
cantidad=[]
prescripcion=[]
tipo=[]
codigo_medicamento=[]
descripcion_medicamento=[]
cantidad_medicamento=[]
prescripcion_medicamento=[]
#presentacion=[]
enes=[]
for n in salidas1['consulta_id'].drop_duplicates():
    if len(salidas1[salidas1['consulta_id']==n])>0:
        cups_solictud1=' '
        descripcion1=' '
        cantidad1=' '
        prescripcion1=' '
        tipo1=' '
        codigo_medicamento1=' '
        descripcion_medicamento1=''
        cantidad_medicamento1=''
        prescripcion_medicamento1=''
        #presentacion1=''
        for k in range(len(salidas1[salidas1['consulta_id']==n])):
            cups_solictud1=cups_solictud1+'-'+str(salidas1[salidas1['consulta_id']==n]['cups_solictud'].to_list()[k])
            descripcion1=descripcion1+'-'+str(salidas1[salidas1['consulta_id']==n]['descripcion'].to_list()[k])
            cantidad1=cantidad1+'-'+str(salidas1[salidas1['consulta_id']==n]['cantidad'].to_list()[k])
            prescripcion1=prescripcion1+'-'+str(salidas1[salidas1['consulta_id']==n]['prescripcion'].to_list()[k])
            tipo1=tipo1+'-'+str(salidas1[salidas1['consulta_id']==n]['tipo'].to_list()[k])
            codigo_medicamento1=codigo_medicamento1+'-'+str(salidas1[salidas1['consulta_id']==n]['codigo_medicamento'].to_list()[k])
            descripcion_medicamento1=descripcion_medicamento1+'-'+str(salidas1[salidas1['consulta_id']==n]['descripcion_medicamento'].to_list()[k])
            cantidad_medicamento1=cantidad_medicamento1+'-'+str(salidas1[salidas1['consulta_id']==n]['cantidad_medicamento'].to_list()[k])
            prescripcion_medicamento1=prescripcion_medicamento1+'-'+str(salidas1[salidas1['consulta_id']==n]['prescripcion_medicamento'].to_list()[k])
            #presentacion1=presentacion1+'-'+str(salidas1[salidas1['consulta_id']==n]['presentacion'].to_list()[k])
        cups_solictud.append(cups_solictud1)
        descripcion.append(descripcion1)
        cantidad.append(cantidad1)
        prescripcion.append(prescripcion1)
        tipo.append(tipo1)
        codigo_medicamento.append(codigo_medicamento1)
        descripcion_medicamento.append(descripcion_medicamento1)
        cantidad_medicamento.append(cantidad_medicamento1)
        prescripcion_medicamento.append(prescripcion_medicamento1)
        #presentacion.append(presentacion1)
        enes.append(n)
    else:
        continue
salidas = pd.DataFrame()
salidas['consulta_id']=enes
salidas['cups_solictud']=cups_solictud
salidas['descripcion']=descripcion
salidas['cantidad']=cantidad
salidas['prescripcion']=prescripcion
salidas['tipo']=tipo
salidas['codigo_medicamento']=codigo_medicamento
salidas['descripcion_medicamento']=descripcion_medicamento
salidas['cantidad_medicamento']=cantidad_medicamento
salidas['prescripcion_medicamento']=prescripcion_medicamento
#salidas['presentacion']=presentacion
for j in entradas['consulta_id'].drop_duplicates():
    if j in enes:
        continue
    else:
        entradas = entradas.drop(entradas[entradas[j]==True].index)
entradas['consulta_id']=entradas['consulta_id'].drop_duplicates().dropna()
entradas=entradas.dropna()
entradas=entradas.sort_values(by=['consulta_id'])
salidas=salidas.sort_values(by=['consulta_id'])
entradas=entradas.drop(columns=['consulta_id'])
salidas=salidas.drop(columns=['consulta_id'])
entradas['T_A'] = entradas['T_A'].str.replace('-','/')
t_a = entradas["T_A"].str.split('/', expand=True)
t_a.columns = ['T_A1', 'T_A2']
entradas = pd.concat([entradas, t_a], axis=1)
entradas['TEST_DE_FINDRISK']=entradas['TEST_DE_FINDRISK'].astype(str).apply(lambda x: x[:2]).str.replace(':',' ').astype(float)
entradas['TASA_DE_FILTRACION_GLOMERULAR']=entradas['TASA_DE_FILTRACION_GLOMERULAR'].astype(str).apply(lambda x: x[:2]).astype(float)
entradas['TEST_DE_FRAMINGHAM']=entradas['TEST_DE_FRAMINGHAM'].astype(str).apply(lambda x: x[:1]).str.replace('%',' ').astype(float)
salidas=salidas.fillna(0)
entradas=entradas.fillna(0)
entradas=entradas.drop(['T_A'], axis=1)
clfx=tree.DecisionTreeClassifier(criterion='gini',max_depth=15,ccp_alpha=0.01,splitter='best')
clfy=tree.DecisionTreeClassifier(criterion='gini',max_depth=15,ccp_alpha=0.01,splitter='best')
tablas_enc = dict()
i=0
for j in entradas.columns.tolist():
    entradas[j]=entradas[j].fillna(0)
    if entradas[j].dtype=='O':
        entradas[j]=entradas[j].astype(str)
        tablas_enc[j] = OrdinalEncoder(categories=[entradas[j].unique().tolist()])
        tablas_enc[j].fit(entradas[[j]])
        entradas[j+'cat']= tablas_enc[j].transform(entradas[[j]])
    else:
        entradas[j+'cat'] = entradas[j]
        continue
for i in salidas.columns.tolist():
    salidas[i]=salidas[i].fillna(0)
    if salidas[i].dtype=='O':
        salidas[i]=salidas[i].astype(str)
        tablas_enc[i]=OrdinalEncoder(categories=[salidas[i].unique().tolist()])
        tablas_enc[i].fit(salidas[[i]])
        salidas[i+'cat'] = tablas_enc[i].transform(salidas[[i]])
    else:
        salidas[i+'cat'] = salidas[i]
        continue
salidas_cat=[]
for k in salidas.columns.tolist():
    if 'cat' in k:
        salidas_cat.append(k)
    else:
        continue
entradas_cat=[]
for h in entradas.columns.tolist():
    if 'cat' in h:
        entradas_cat.append(h)
    else:
        continue
salidas_cat1=[]
salidas_cat1.append('cups_solictudcat')
salidas_cat1.append('descripcioncat')
salidas_cat1.append('cantidadcat')
salidas_cat1.append('prescripcioncat')
salidas_cat1.append('tipocat')
salidas_cat2=[]
salidas_cat2.append('codigo_medicamentocat')
salidas_cat2.append('descripcion_medicamentocat')
salidas_cat2.append('cantidad_medicamentocat',)
salidas_cat2.append('prescripcion_medicamentocat')
Y1=salidas[salidas_cat1]
X1=entradas[entradas_cat]
Y2=salidas[salidas_cat2]
X2=salidas[salidas_cat1]
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=0.20)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y2, test_size=0.20)

clf1=clfx.fit(X_train1, y_train1)
clf2=clfy.fit(X_train2, y_train2)
class Item(BaseModel):
  Edadcat: str
  sexocat: str
  grupo_sanguineocat: str
  Pesocat: str
  Tallacat: str
  Tempcat: str
  F_Ccat: str
  F_Rcat: str
  TEST_DE_FINDRISKcat: str
  TASA_DE_FILTRACION_GLOMERULARcat: str
  TEST_DE_FRAMINGHAMcat: str
  diagnostico_principalcat: str
  #diagnostico_secundariocat: str
  #diagnostico_terciariocat: str
  #cuarto_diagnosticocat: str
  T_a1cat:str
  T_a2cat: str



@app.post("/")
async def root(item: Item):
    dato_en=[]
    dato_en.append(int(item.Edadcat))
    dato_en.append(tablas_enc['sexo'].transform([[item.sexocat]])[0][0])
    dato_en.append(float(item.Tallacat))
    dato_en.append(tablas_enc['grupo_sanguineo'].fit_transform([[item.grupo_sanguineocat]])[0][0])
    dato_en.append(float(item.Pesocat))
    dato_en.append(float(item.Tempcat))
    dato_en.append(int(item.F_Ccat))#F_C
    dato_en.append(int(item.F_Rcat))#F_R
    dato_en.append(float(item.TEST_DE_FINDRISKcat))#Test de frindrisk
    dato_en.append(float(item.TASA_DE_FILTRACION_GLOMERULARcat))#Tasa de filtracion
    dato_en.append(float(item.TEST_DE_FRAMINGHAMcat))#Test de framingan ,porcentaje
    dato_en.append(tablas_enc['diagnostico_principal'].fit_transform([[item.diagnostico_principalcat]])[0][0])#diagnostico_principal
    #dato_en.append(tablas_enc['diagnostico_secundario'].fit_transform([[item.diagnostico_secundariocat]]))#diagnostico_secundario
    #dato_en.append(tablas_enc['diagnostico_terciario'].fit_transform([[item.diagnostico_terciariocat]]))#diagnostico_terciario
    #dato_en.append(tablas_enc['cuarto_diagnostico'].fit_transform([[item.cuarto_diagnosticocat]]))#cuarto_diagnostico
    dato_en.append(int(item.T_a1cat))#T_a1
    dato_en.append(int(item.T_a2cat))#T_a2
    prediccion= clf1.predict([dato_en])
    lists = prediccion.tolist()
    prediccion2= clf2.predict(lists)
    lists2 = prediccion2.tolist()
    respuesta3=0
    respuesta3={
        "cups_solictud":tablas_enc['cups_solictud'].inverse_transform([[prediccion[0][0]]])[0][0],
        "descripcion" :tablas_enc['descripcion'].inverse_transform([[prediccion[0][1]]])[0][0],
        "cantidad":tablas_enc['cantidad'].inverse_transform([[prediccion[0][2]]])[0][0],
        "prescripcion": tablas_enc['prescripcion'].inverse_transform([[prediccion[0][3]]])[0][0][0][0],
        "tipo": tablas_enc['tipo'].inverse_transform([[prediccion[0][4]]])[0][0],
        "codigo_medicamento":tablas_enc['codigo_medicamento'].inverse_transform([[prediccion2[0][0]]])[0][0],
        "descripcion_medicamento":tablas_enc['descripcion_medicamento'].inverse_transform([[prediccion2[0][1]]])[0][0],
        "cantidad_medicamento":tablas_enc['cantidad_medicamento'].inverse_transform([[prediccion2[0][2]]])[0][0],
        "prescripcion_medicamento":tablas_enc['prescripcion_medicamento'].inverse_transform([[prediccion2[0][3]]])[0][0],
        #"presentacion":tablas_enc['presentacion'].inverse_transform([[prediccion2[0][4]]])[0][0]
        }
    return respuesta3