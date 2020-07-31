# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render, render_to_response, redirect
from django.template import loader
from django.http import HttpResponse
import pyopencl as cl
import pyopencl.algorithm as algo
import pyopencl.array as pyci_array
import numpy as np
import pandas as pd
from .models import Post
from django import forms
import os
from django.core.exceptions import ValidationError
import Teuvonet_Jetson_Site.settings as settings
from .main_host_Classification import the_entire_process_classification
import main_host_Classification
from .main import entire_process
from django.views.generic.edit import CreateView


#import pycuda.driver as cuda
# Create your views here.

global balanced_data;

class NewExperimentForm(forms.ModelForm):
	class Meta:
		model = Post
		fields = '__all__'
		widgets = {
			    'dataset_name': forms.TextInput(attrs={'class': 'white-text'}),
			}
	def clean(self):
		cleaned_data=super(NewExperimentForm, self).clean()
		algorithm_choice=cleaned_data.get("selected_algo")
		dataset_name  = cleaned_data.get("dataset_name")
		
		if algorithm_choice:
			pass#print('Here Algo')
		else:
			raise forms.ValidationError(('Need to select method of learning.<script>alert("Hello world");</script>'))	
		if dataset_name:
			pass#print('Here Algo')
		else:
			raise forms.ValidationError(('Need to select data_set name..<script>alert("Hello world");</script>'))
															
		return cleaned_data				

class NewExperiment(CreateView):
	model = Post
	form_class=NewExperimentForm
	template_name='upload.html'
	#success_url='/result/'	
	
		

	def form_invalid(self, form, **kwargs):
		context = self.get_context_data(**kwargs)
		context['form'] = form
		# here you can add things like:
	
	def form_valid(self, form, **kwargs):
		
		context = self.get_context_data(**kwargs)
		dataset_name=form.cleaned_data['dataset_name']
		selected_algo = form.cleaned_data['selected_algo']
		
		if selected_algo == None:
			print("No Algo selected")

		print("Selected Algorithm : "+str(selected_algo))
		#print(target)
		#print(inputfile)
		self.object=form.save()
		
			
		context_django = {}

                print("dataset_name:"+str(dataset_name))
		#context['dataset_name'] = os.path.basename(my_file)
		#context['target_name'] = target

		try:
			#print("Try Block")
			platforms = cl.get_platforms()
			#print(platforms[1])
			devices = platforms[1].get_devices()
			#print(devices[0])
			ctx = cl.Context([devices[0]])
		except:
			print("ERROR")
			exit()
		try:	
			if selected_algo == 'Function Approximation':
			
			
				the_entire_process(input_data_df, target, should_i_boost, test_data_orig_df, my_context)
			
				context.update(my_context)

				try:
					print("Context Pop")
					print(type(ctx))
					ctx.pop()

				except:
					print("Somethings wrong ctx")


				return render_to_response('result.html', context)
			elif selected_algo == 'Classification':
				print("dataset_name",dataset_name)
				accuracy_score = entire_process(dataset_name, context)
								
				try:
					ctx.pop()
				except:
					print("Somethings wrong")

				print("accuracy_score",accuracy_score)			
				print("This was in Django")
				return render_to_response('result_classification.html', context)
		except KeyError:
			context['target_error'] ="Target Selected is not in the CSV file or Wrong Algorithm Selected. System will reload in a few seconds"
			return render_to_response('upload.html', context)
		except ValueError:
			context['target_error'] ="Target Selected is not in the CSV file or Wrong Algorithm Selected. System will reload in a few seconds"
			return render_to_response('upload.html', context)				
		except IndexError:
			context['target_error'] ="Target Selected is not in the CSV file or Wrong Algorithm Selected. System will reload in a few seconds"
			return render_to_response('upload.html', context)

					

	def post(self, request, *args, **kwargs):
		self.object=None
		form_class = self.get_form_class()
		form = self.get_form(form_class)
		#print('Form post')
		if form.is_valid():
			print('Form valid')
			return self.form_valid(form, **kwargs)
		else:
			print('Form invalid')
		return self.form_invalid(form, **kwargs) 



def home(request):
    template = loader.get_template('note.html')
    context = {}

    context['platform_name'] = "Dummy_platform"#platforms[0]
    context['device_name'] = "Dummy_device"#platforms[0].get_devices()[0]
    context['dataset_name'] = "Dummy"
    context['target_name'] = "Dummy_target"

    return render(request, 'note.html', context)
    # return render_to_response("note.html", notes)

def result(request):
    context = {}
    #main_host.main(context)
    return render(request, 'result.html', context)
    
def failure(request):
    context = {}
    #main_host.main(context)
    return render(request, 'upload_failure.html', context)
def result_classification(request):
    context = {}
    #main_host.main(context)
    return render(request, 'result_classification.html', context)

def blank(request):
    y1 ="Waiting for results !!!"
    context = {'y1': y1}
    return render(request, 'blank.html', context)
