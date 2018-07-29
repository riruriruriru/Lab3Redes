import numpy as np
from numpy import sin, linspace, pi
from scipy.io.wavfile import read, write
from scipy.io import wavfile
from scipy import fft, arange, ifft
from scipy import signal
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy.interpolate import interp1d
import scipy.integrate as integrate

import matplotlib.pyplot as plt
def menu():
    #Muestra el menu principal y usando input se pide al usuario que ingrese la opcion que desea ejecutar
    option = 0
    while option == 0:
        print('Menu Principal')
        print('Opciones:')
        print('1) Ingresar nombre de archivo de audio')
        print('2) Salir')
        user_input = input('Ingrese el numero de la opcion que desea ejecutar: ')
        if user_input=="2":
            #se retorna y finaliza el programa
            return 0
        elif user_input=="1":
            #se pide ingresar un nombre de archivo
            error = 1
            while error == 1:
                input_nombre = input('Ingrese nombre del archivo de audio: ')
                #la apertura se realiza inicialmente en un try-except, para evitar que el programa se caiga
                #en caso de que el archivo no pueda ser abierto
                try:
                    open(input_nombre, 'rb')
                    error = 0
                except FileNotFoundError:
                    error = 1       
                    print('nombre de archivo no existente o fuera de directorio')
            #llamado a funcion que abre un archivo verificado y retorna datos de este
            rate, info, data , timp, t = process_audio(input_nombre)
            option = second_menu(rate, info, data, timp, t)
            
        else:
            print('Ingrese una opcion correcta')
            
    return
            
def second_menu(rate, info, data, timp, t):
    option = 0
    #Segundo menu que se llama cuando se abre correctamente un archivo
    while option == 0:
        print('##########################')
        print('Menu Archivo')
        print('Opciones:')
        print('1) Modulacion FM analoga')
        print('2) Modulacion AM analoga')
        print('3) Retroceder al menu anterior')
        print('4) Salir')
        user_input = input('Ingrese el numero de la opcion que desea ejecutar: ')
        if user_input=="3":
            #opcion que sirve para retroceder al menu anterior y abrir un nuevo archivo
            return 0
        elif user_input=="1":            
            #funcion que el espectrograma
            spec, freq, time, im = obtainSpectrogram(rate, data)
            option = third_menu(rate, data, freq, spec, time, im, t, info)
        elif user_input=="2":
            spec, freq, time, im = obtainSpectrogram(rate, data)
            option = third_menu_am(rate, data, freq, spec, time, im, t, info)
        elif user_input=="4":
            
            return 2
        else:
            print("Ingrese una opcion correcta")
    return


def third_menu(rate, data, freq, spec, time, im, t, info):
    option = 0
    while option == 0:
        print('####################')
        print('Menu FM')
        print('Opciones:')
        print('1) Aplicar modulacion analoga al 15%')
        print('2) Aplicar modulacion analoga al 100%')
        print('3) Aplicar modulacion analoga al 125%')
        print('4) Retroceder al menu anterior')
        print('5) Salir')
        user_input = input('Ingrese el numero de la opcion que desea ejecutar: ')
        if user_input=="1":
            FM_analog_modulation(rate, data, time, 0.15, t, info)
            #option = fourth_menu(rate, f_data, f_data2, f_data3, "Low Pass")
        elif user_input=="2":
            FM_analog_modulation(rate, data, time, 1, t, info)
            #option = fourth_menu(rate, f_data, f_data2, f_data3, "High Pass")                        
        elif user_input=="3":
            FM_analog_modulation(rate, data, time, 1.25, t, info)
            #option = fourth_menu(rate, f_data, f_data2, f_data3, "Band Pass")                 
        elif user_input=="4":
            return 0
        elif user_input=="5":
            return 2
        else:
            print('Ingrese una opcion valida')
    return
def third_menu_am(rate, data, freq, spec, time, im, t, info):
    option = 0
    while option == 0:
        print('####################')
        print('Menu AM')
        print('Opciones:')
        print('1) Aplicar modulacion analoga al 15%')
        print('2) Aplicar modulacion analoga al 100%')
        print('3) Aplicar modulacion analoga al 125%')
        print('4) Retroceder al menu anterior')
        print('5) Salir')
        user_input = input('Ingrese el numero de la opcion que desea ejecutar: ')
        if user_input=="1":
            fc,  newTime, resultado, beta, data2 = AM_analog_modulation(rate, data, time, 0.15, t, info)
            AM_demodulation_menu(fc,  newTime, resultado, beta, data2, rate, info)
            #option = fourth_menu(rate, f_data, f_data2, f_data3, "Low Pass")
        elif user_input=="2":
            fc,  newTime, resultado, beta, data2 = AM_analog_modulation(rate, data, time, 1, t, info)
            AM_demodulation_menu(fc,  newTime, resultado, beta, data2, rate, info)
            #option = fourth_menu(rate, f_data, f_data2, f_data3, "High Pass")                        
        elif user_input=="3":
            fc,  newTime, resultado, beta, data2 = AM_analog_modulation(rate, data, time, 1.25, t, info)
            AM_demodulation_menu(fc,  newTime, resultado, beta, data2, rate, info)
            #option = fourth_menu(rate, f_data, f_data2, f_data3, "Band Pass")                 
        elif user_input=="4":
            return 0
        elif user_input=="5":
            return 2
        else:
            print('Ingrese una opcion valida')
    return

#Filtro de tipo fir low pass
#Recibe los datos retornados por obtainspectrogram
def firLowPass(rate, data, t, info, fc):
    #se calcula la frecuencia de nyquist 
    nyq_f = rate/2.0
    cutoff =  nyq_f*0.09
    numtaps = 1001
    print("###########################")
    print("nyq: ")
    print(nyq_f)
    #se obtienen los valores que se usaran para filtrar con firwin
    print("cutoff: ")
    print(cutoff/nyq_f)
    print("#########")
    #taps = signal.firwin(numtaps, cutoff=cutoff, nyq = nyq_f, window = 'hamming')
    taps = signal.firwin(numtaps, 0.9, window = 'hamming')
    #se aplica el filtro con lfilter
    t2 = linspace(0,len(data)/(rate),len(data))
    y = signal.lfilter(taps, 1.0, data)
    #se grafican los tres filtros
    graph(t2, y, "Tiempo[s]", "Amplitud[db]","Filtro Low-Pass señal demodulada")
    freq, fourierT = fourier(rate/10, info, y)
    graph(freq, fourierT, "Frecuencia[hz]", "Magnitud de Frecuencia[db]","Transformada de Fourier: filtro Low-Pass señal demodulada")

    #se retornan los valores de las amplitudes
    return y


def AM_demodulation_menu(fc,  newTime, resultado, beta, data, rate, info):
	option = 0
	text = str(beta*100) + "%"
	while option == 0:
		print('####################')
		print('Menu AM Demodulacion')
		print('Opciones:')
		print('1) Aplicar demodulacion analoga AM al: ' + text)
		print('2) Retroceder al menu anterior')
		print('3) Salir')
		user_input = input('Ingrese el numero de la opcion que desea ejecutar: ')
		if user_input=="1":
			AM_demodulation(data, rate, fc, newTime, beta, info)
		elif user_input=="2":
			return 0                        
		elif user_input=="3":
			return 2
		else:
			print('Ingrese una opcion valida')
	return
	
	

def fourth_menu(rate, f_data, f_data2, f_data3, nombre):
    option = 0
    while option == 0:
        print('#####################')
        print('Menu Filtro Tipo ' + str(nombre))
        print('Opciones:')
        print('1) Mostrar Espectrograma')
        print('2) Guardar archivo de audio filtrado .wav')
        print('3) Retroceder al menu anterior')
        print('4) Salir')
        user_input = input('Ingrese el numero de la opcion que desea ejecutar: ')
        if user_input=="1":
            print("Ventana Hamming")
            obtainSpectrogram(rate, f_data)
            print("Ventana Nutall")
            obtainSpectrogram(rate, f_data2)
            print("Ventana Blackman")
            obtainSpectrogram(rate, f_data3)            
        elif user_input=="2":
            if nombre == "High Pass":
                name1 = "beaconHighPassH.wav"
                name2 = "beaconHighPassN.wav"
                name3 = "beaconHighPassB.wav"
            elif nombre == "Low Pass":
                name1 = "beaconLowPassH.wav"
                name2 = "beaconLowPassN.wav"
                name3 = "beaconLowPassB.wav"
            elif nombre == "Band Pass":
                name1 = "beaconBandPassH.wav"
                name2 = "beaconBandPassN.wav"
                name3 = "beaconBandPassB.wav"                
            else:
                print("caso invalido")
            write(name1, rate, f_data)
            write(name2,rate, f_data2)
            write(name3, rate, f_data3)
            print("Archivo guardado con nombre: " + str(name1))
            print("Archivo guardado con nombre: " + str(name2))
            print("Archivo guardado con nombre: " + str(name3))
            
        elif user_input == "3":
            return 0
        elif user_input=="4":
            return 2
        else:
            print('Ingrese una opcion valido')
    return 0

#Muestra espectrograma con los datos del archivo de audio leido
#Se utiliza la funcion de plt "specgram"
def obtainSpectrogram(rate, data):
    spectrogram, freq, time, im = plt.specgram(data, NFFT=1024, Fs = rate)
    plt.xlabel('Tiempo[s]')
    plt.ylabel('Frecuencia[Hz]')
    plt.show()
    return spectrogram, freq, time, im
#funcion que se encarga de recibir un audio y obtener sus datos
def process_audio(archivo):
    
    rate,info=wavfile.read(archivo)
    print('rate')
    print(rate)
    print(info)
    dimension = info[0].size
    print(dimension)
    if dimension==1:
        data = info
        perfect = 1
    else:
        data = info[:,dimension-1]
        perfect = 0
    timp = len(data)/rate
    t=linspace(0,timp,len(data))
    print("LARGO t")
    print(len(t))    
    return rate, info, data, timp, t
#funcion que recibe dos arreglos de la misma dimension y los grafica con las etiquetas tambien recibidas por argumento
def graph(x, y, labelx, labely, title):
    plt.title(title)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.plot(x, y)
    plt.show()
    return

def fourier(rate, info, data):
    #funcion que transforma los datos obtenidos del archivo de sonidos al dominio de las frecuencias usando la transformada de fourier (rfft)
    timp = len(data)/rate
    large = len(data)
    #utilizando timp, se obtiene el tiempo total del archivo
    fourierTransform = np.fft.fftshift(fft(data))
    k = linspace(-len(fourierTransform)/2, len(fourierTransform)/2, len(fourierTransform))
    frq = k/timp
    print('largo de frecuencias ' + str(len(frq))+'largo datos ' + str(len(data)))
    #con k se obtiene un arreglo con tamanio igual al largo de los datos obtenidos con la transformada y se dividen por el tiempo, obteniendo un arreglo de frecuencias
    print(fourierTransform)   
    #se retorna el arreglo de frecuencias y datos de la transformada de fourier
    
    return frq, fourierTransform

def FM_analog_modulation(rate, data, time, beta, t, info):
	'''fc = 3000
	fm = 220
	output = np.cos(np.pi*2*fc*t*(beta)+data)
	'''
	title = str(beta*100) +"%"
	data2 = interpolate(t, data, rate, info)
	newLen = len(data2)
	newTime = np.linspace(0,len(data)/(rate), newLen)
	#t2_fm = np.linspace(0,T, 400000*T)
	#data_fm = np.interp(t2_fm, t1, data)
	#t3_fm = np.linspace(0, 400000, 400000*T)
	A = 1
	k = 0.15
	carrier = np.sin(2*np.pi*newTime);
	w = rate*newTime
	integral = integrate.cumtrapz(data2, newTime, initial=0)
	resultado = np.cos(np.pi*w + integral*np.pi);
	plt.plot(newTime[1000:4000], resultado[1000:4000])
	plt.show()
	#grafico normal
	graph(t, data, "Tiempo[s]", "Amplitud[db]", "Tiempo vs Amplitud")
	#grafico modulado
	graph(newTime[1000:4000], resultado[1000:4000], "Tiempo[s]", "Amplitud[db]", "Modulacion FM al " +title)
	#grafico fourier
	freq, fourierT = fourier(rate*10, info, data)
	graph(freq, fourierT, "Frecuencia[hz]", "Magnitud de Frecuencia[db]", "Transformada de Fourier datos originales")
	#grafico fourier modulado
	#se grafica Frecuencia vs Magnitud
	freq2, fourierT2 = fourier(rate*10, info, resultado)
	graph(freq2, fourierT2, "tiempo", "frecuencia", "Transformada de fourier modulacion FM al " +title)
	return
	
def interpolate(t, data, rate, info):
    interp = interp1d(t,data)
    newTime = np.linspace(0,len(data)/rate,len(data)*10)
    resultado = interp(newTime)
    return resultado
	
def AM_analog_modulation(rate,data,time, beta,t, info):
    title = str(beta*100) +"%"
    print("RATE: ")
    print(rate)
    data2 = interpolate(t, data, rate, info)
    fc=20000
    newLen = len(data2)
    newTime = np.linspace(0,len(data)/(rate), newLen)
    carrier = np.cos(2*np.pi*newTime*fc)*beta
    resultado = carrier*data2
    plt.plot(newTime[1000:2000], resultado[1000:2000])
    plt.show()
    #grafico normal sin resample
    graph(t, data, "Tiempo[s]", "Amplitud [db]", "Tiempo vs Amplitud")
    #grafico modulado con resample
    graph(newTime[1000:2000], resultado[1000:2000], "Tiempo[s]", "Amplitud [db]", "Tiempo vs Amplitud Modulado AM al "+title)
    #grafico normal fourier sin resample
    freqO, fourierTO = fourier(rate, info, data)
    graph(freqO, fourierTO, "Frecuencia[hz]", "Magnitud de Frecuencia[db]", "Frecuencia vs Magnitud de Frecuencia datos originales")
    #fourier resample
    freq, fourierT = fourier(rate*10, info, data2)
    #se grafica Frecuencia vs Magnitud
    graph(freq, fourierT, "Frecuencia[hz]", "Magnitud de Frecuencia[db]", "Frecuencia vs Magnitud de Frecuencia datos originales")
    #grafico fourier modulado
    freq2, fourierT2 = fourier(rate*10, info, resultado)
    #se grafica Frecuencia vs Magnitud
    graph(freq2, fourierT2, "Frecuencia[hz]", "Magnitud de Frecuencia[db]", "Frecuencia vs Magnitud de Frecuencia Modulado AM al " + title)
    return fc,  newTime, resultado, beta, data2



def AM_demodulation(data,rate,fc, newTime, beta, info):
    resultado = data*np.cos(2*np.pi*fc*newTime)
    #grafico demodulado
    graph(newTime, resultado, "Tiempo[s]", "Amplitud [db]", "Tiempo vs Amplitud demodulado AM")
    #grafico demodulado fourier
    freq, fourierT = fourier(rate*10, info, resultado)
    #se grafica Frecuencia vs Magnitud
    graph(freq, fourierT, "Frecuencia[hz]", "Magnitud de Frecuencia[db]", "Frecuencia vs Magnitud de Frecuencia demodulado AM")
    firLowPass(10*rate, resultado, newTime, info, fc)
    return resultado


menu()

