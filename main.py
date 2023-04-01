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
from joblib import load

app = FastAPI()


tablas_enc = load('tables.joblib')
clf1 = load('clf1.joblib')
clf2 = load('clf2.joblib')
tablas_enc2 = load('tables2.joblib')
clf11 = load('clf11.joblib')
clf22 = load('clf22.joblib')


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
  perimetro: str
  condicion: str
  sp02: str

@app.post("/")
async def root(item: Item):
    if item.condicion=='2':
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
        class MiClase:
          def __init__(self, cups_solicitud, descripcion,cantidad,tipo):
            self.cups_solicitud = cups_solicitud
            self.descripcion = descripcion
            self.cantidad= cantidad
            self.tipo= tipo

        class MiClase2:
          def __init__(self, codigo_medicamento, descripcion_medicamento,cantidad_medicamento,prescripcion_medicamento):
            self.codigo_medicamento = codigo_medicamento
            self.descripcion_medicamento = descripcion_medicamento
            self.cantidad_medicamento= cantidad_medicamento
            self.prescripcion_medicamento= prescripcion_medicamento

        mi_diccionario = {
           "cups_solictud":tablas_enc['cups_solictud'].inverse_transform([[prediccion[0][0]]])[0][0].replace('-0.0','').split('-')[1:],
           "descripcion" :tablas_enc['descripcion'].inverse_transform([[prediccion[0][1]]])[0][0].replace('-0.0','').split('-')[1:],
           "cantidad":tablas_enc['cantidad'].inverse_transform([[prediccion[0][2]]])[0][0].replace('-0.0','').split('-')[1:],
           "tipo": tablas_enc['tipo'].inverse_transform([[prediccion[0][4]]])[0][0].replace('-0.0','').split('-')[1:],
        }

        mi_diccionario2 = {
           "codigo_medicamento":tablas_enc['codigo_medicamento'].inverse_transform([[prediccion2[0][0]]])[0][0].replace('-0.0','').split('-')[1:],
           "cantidad_medicamento":tablas_enc['cantidad_medicamento'].inverse_transform([[prediccion2[0][2]]])[0][0].replace('-0.0','').split('-')[1:],
           "descripcion_medicamento":tablas_enc['descripcion_medicamento'].inverse_transform([[prediccion2[0][1]]])[0][0].replace('-0.0','').split('-')[1:],
           "prescripcion_medicamento":tablas_enc['prescripcion_medicamento'].inverse_transform([[prediccion2[0][3]]])[0][0].replace('-0.0','').split('-')[1:],
        }

        solicitudes=[]
        medicamentos=[]

        dist=len(mi_diccionario['cups_solictud']),len(mi_diccionario['descripcion']),len(mi_diccionario['cantidad']),len(mi_diccionario['tipo'])
        dist2=len(mi_diccionario2['codigo_medicamento']),len(mi_diccionario2['descripcion_medicamento']),len(mi_diccionario2['cantidad_medicamento']),len(mi_diccionario2['prescripcion_medicamento'])

        for i in range(min(dist)):
          cups_solicitud = mi_diccionario['cups_solictud'][i]
          descripcion = mi_diccionario['descripcion'][i]
          cantidad = mi_diccionario['cantidad'][i]
          tipo = mi_diccionario['tipo'][i]
          solicitudes.append(MiClase(cups_solicitud, descripcion,cantidad,tipo))

        for j in range(min(dist2)):
          codigo_medicamento = mi_diccionario2['codigo_medicamento'][j]
          descripcion_medicamento = mi_diccionario2['descripcion_medicamento'][j]
          cantidad_medicamento = mi_diccionario2['cantidad_medicamento'][j]
          prescripcion_medicamento = mi_diccionario2['prescripcion_medicamento'][j]
          medicamentos.append(MiClase2(codigo_medicamento, descripcion_medicamento,cantidad_medicamento,prescripcion_medicamento))
        respuesta3=0
        respuesta3={
            "clase_solicitudes": solicitudes,
            "clase_medicamentos": medicamentos,
            "cups_solicitud_probabilidad":max(clf1.predict_proba([dato_en])[0][0]),
            "descripcion_probabilidad":max(clf1.predict_proba([dato_en])[1][0]),
            "cantidad_probabilidad":max(clf1.predict_proba([dato_en])[2][0]),
            "tipo_probabilidad":max(clf1.predict_proba([dato_en])[4][0]),
            "codigo_medicamento_probabilidad":max(clf2.predict_proba(lists)[0][0]),
            "descripcion_medicamento_probabilidad":max(clf2.predict_proba(lists)[1][0]),
            "cantidad_medicamento_probabilidad":max(clf2.predict_proba(lists)[2][0]),
            "prescripcion_medicamento_probabilidad":max(clf2.predict_proba(lists)[3][0]),
            #"presentacion":tablas_enc['presentacion'].inverse_transform([[prediccion2[0][4]]])[0][0]
            }
    elif item.condicion=='1':
        dato_en=[]
        dato_en.append(int(item.Edadcat)) #Edad
        dato_en.append(tablas_enc2['sexo'].transform([[item.sexocat]])[0][0]) #sexo
        dato_en.append(tablas_enc2['grupo_sanguineo'].fit_transform([[item.grupo_sanguineocat]])[0][0])#grupo_sanguineo
        dato_en.append(float(item.Pesocat)) #peso 
        dato_en.append(float(item.Tallacat)) #talla
        dato_en.append(float(item.Tempcat)) #temperatura
        dato_en.append(int(item.F_Ccat))#F_C 
        dato_en.append(float(item.perimetro))#perimetro cefalico
        dato_en.append(float(item.F_Rcat))#frecuencia respiratoria
        dato_en.append(float(item.sp02))#spO2
        dato_en.append(tablas_enc2['diagnostico_principal'].fit_transform([[item.diagnostico_principalcat]])[0][0])#diagnostico_principal
        #dato_en.append(tablas_enc['diagnostico_secundario'].fit_transform([[item.diagnostico_secundariocat]]))#diagnostico_secundario
        #dato_en.append(tablas_enc['diagnostico_terciario'].fit_transform([[item.diagnostico_terciariocat]]))#diagnostico_terciario
        #dato_en.append(tablas_enc['cuarto_diagnostico'].fit_transform([[item.cuarto_diagnosticocat]]))#cuarto_diagnostico
        dato_en.append(float(item.T_a1cat))#T_a1
        dato_en.append(float(item.T_a2cat))#T_a2
        prediccion= clf11.predict([dato_en])
        lists = prediccion.tolist()
        prediccion2= clf22.predict(lists)
        lists2 = prediccion2.tolist()
        class MiClase:
           def __init__(self, cups_solicitud, descripcion,cantidad,tipo):
              self.cups_solicitud = cups_solicitud
              self.descripcion = descripcion
              self.cantidad= cantidad
              self.tipo= tipo

        class MiClase2:
            def __init__(self, codigo_medicamento, descripcion_medicamento,cantidad_medicamento,prescripcion_medicamento):
              self.codigo_medicamento = codigo_medicamento
              self.descripcion_medicamento = descripcion_medicamento
              self.cantidad_medicamento= cantidad_medicamento
              self.prescripcion_medicamento= prescripcion_medicamento

        mi_diccionario = {
           'cups_solictud': tablas_enc2['cups_codigo'].inverse_transform([[prediccion[0][0]]])[0][0].replace('-0.0','').split('-')[1:],
           "descripcion" :tablas_enc2['descripcion'].inverse_transform([[prediccion[0][1]]])[0][0].replace('-0.0','').split('-')[1:],
           "cantidad":tablas_enc2['cantidad'].inverse_transform([[prediccion[0][2]]])[0][0].replace('-0.0','').split('-')[1:],
           "tipo": tablas_enc2['tipo'].inverse_transform([[prediccion[0][4]]])[0][0].replace('-0.0','').split('-')[1:],
        }

        mi_diccionario2 = {
           "codigo_medicamento":tablas_enc2['codigo_medicamento'].inverse_transform([[prediccion2[0][0]]])[0][0].replace('-0.0','').split('-')[1:],
           "descripcion_medicamento":tablas_enc2['descripcion_medicamento'].inverse_transform([[prediccion2[0][1]]])[0][0].replace('-0.0','').split('-')[1:],
           "cantidad_medicamento":tablas_enc2['cantidad_medicamento'].inverse_transform([[prediccion2[0][2]]])[0][0].replace('-0.0','').split('-')[1:],
           "prescripcion_medicamento":tablas_enc2['prescripcion_medicamento'].inverse_transform([[prediccion2[0][3]]])[0][0].replace('-0.0','').split('-')[1:],
        }

        solicitudes=[]
        medicamentos=[]

        dist=len(mi_diccionario['cups_solictud']),len(mi_diccionario['descripcion']),len(mi_diccionario['cantidad']),len(mi_diccionario['tipo'])
        dist2=len(mi_diccionario2['codigo_medicamento']),len(mi_diccionario2['descripcion_medicamento']),len(mi_diccionario2['cantidad_medicamento']),len(mi_diccionario2['prescripcion_medicamento'])

        for i in range(min(dist)):
           cups_solicitud = mi_diccionario['cups_solictud'][i]
           descripcion = mi_diccionario['descripcion'][i]
           cantidad = mi_diccionario['cantidad'][i]
           tipo = mi_diccionario['tipo'][i]
           solicitudes.append(MiClase(cups_solicitud, descripcion,cantidad,tipo))

        for j in range(min(dist2)):
           codigo_medicamento = mi_diccionario2['codigo_medicamento'][j]
           descripcion_medicamento = mi_diccionario2['descripcion_medicamento'][j]
           cantidad_medicamento = mi_diccionario2['cantidad_medicamento'][j]
           prescripcion_medicamento = mi_diccionario2['prescripcion_medicamento'][j]
           medicamentos.append(MiClase2(codigo_medicamento, descripcion_medicamento,cantidad_medicamento,prescripcion_medicamento))
        respuesta3=0
        respuesta3={
              "clase_solicitudes": solicitudes,
              "clase_medicamentos": medicamentos,
              "cups_codigo_probabilidad":max(clf11.predict_proba([dato_en])[0][0]),
              "descripcion_probabilidad":max(clf11.predict_proba([dato_en])[1][0]),
              "cantidad_probabilidad":max(clf11.predict_proba([dato_en])[2][0]),
              "tipo_probabilidad":max(clf11.predict_proba([dato_en])[4][0]),
              "codigo_medicamento_probabilidad":max(clf22.predict_proba(lists)[0][0]),
              "descripcion_medicamento_probabilidad":max(clf22.predict_proba(lists)[1][0]),
              "cantidad_medicamento_probabilidad":max(clf22.predict_proba(lists)[2][0]),
              "prescripcion_medicamento_probabilidad":max(clf22.predict_proba(lists)[3][0]),
              #"presentacion":tablas_enc['presentacion'].inverse_transform([[prediccion2[0][4]]])[0][0]
          }
    return respuesta3