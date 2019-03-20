Search.setIndex({docnames:["advanced/config","advanced/dataset","advanced/hook","advanced/index","advanced/main_loop","advanced/model","advanced/python","cli","emloop/emloop","emloop/emloop.constants","emloop/emloop.datasets","emloop/emloop.hooks","emloop/emloop.models","emloop/index","getting_started","index","tutorial"],envversion:55,filenames:["advanced/config.rst","advanced/dataset.rst","advanced/hook.rst","advanced/index.rst","advanced/main_loop.rst","advanced/model.rst","advanced/python.rst","cli.rst","emloop/emloop.rst","emloop/emloop.constants.rst","emloop/emloop.datasets.rst","emloop/emloop.hooks.rst","emloop/emloop.models.rst","emloop/index.rst","getting_started.rst","index.rst","tutorial.rst"],objects:{"":{emloop:[8,0,0,"-"]},"emloop.AbstractDataset":{__init__:[8,2,1,""]},"emloop.AbstractHook":{__init__:[8,2,1,""],after_batch:[8,2,1,""],after_epoch:[8,2,1,""],after_epoch_profile:[8,2,1,""],after_training:[8,2,1,""],before_training:[8,2,1,""],register_mainloop:[8,2,1,""]},"emloop.AbstractModel":{__init__:[8,2,1,""],input_names:[8,3,1,""],output_names:[8,3,1,""],run:[8,2,1,""],save:[8,2,1,""]},"emloop.BaseDataset":{__init__:[8,2,1,""],_configure_dataset:[8,2,1,""],stream_info:[8,2,1,""]},"emloop.DownloadableDataset":{_configure_dataset:[8,2,1,""],data_root:[8,3,1,""],download:[8,2,1,""],download_urls:[8,3,1,""]},"emloop.MainLoop":{EMPTY_ACTIONS:[8,3,1,""],INCORRECT_CONFIG_ACTIONS:[8,3,1,""],UNUSED_SOURCE_ACTIONS:[8,3,1,""],__enter__:[8,2,1,""],__exit__:[8,2,1,""],__init__:[8,2,1,""],_check_sources:[8,2,1,""],_epoch_impl:[8,2,1,""],_run_epoch:[8,2,1,""],epoch:[8,2,1,""],extra_streams:[8,3,1,""],fixed_epoch_size:[8,3,1,""],get_stream:[8,2,1,""],prepare_streams:[8,2,1,""],run_evaluation:[8,2,1,""],run_training:[8,2,1,""],training_epochs_done:[8,3,1,""]},"emloop.constants":{EL_BUFFER_SLEEP:[9,4,1,""],EL_CONFIG_FILE:[9,4,1,""],EL_DEFAULT_LOG_DIR:[9,4,1,""],EL_DEFAULT_TRAIN_STREAM:[9,4,1,""],EL_FULL_DATE_FORMAT:[9,4,1,""],EL_HOOKS_MODULE:[9,4,1,""],EL_LOG_DATE_FORMAT:[9,4,1,""],EL_LOG_FILE:[9,4,1,""],EL_LOG_FORMAT:[9,4,1,""],EL_NA_STR:[9,4,1,""],EL_PREDICT_STREAM:[9,4,1,""],EL_TRACE_FILE:[9,4,1,""]},"emloop.datasets":{AbstractDataset:[10,1,1,""],BaseDataset:[10,1,1,""],DownloadableDataset:[10,1,1,""],StreamWrapper:[10,1,1,""]},"emloop.datasets.AbstractDataset":{__init__:[10,2,1,""]},"emloop.datasets.BaseDataset":{__init__:[10,2,1,""],_configure_dataset:[10,2,1,""],stream_info:[10,2,1,""]},"emloop.datasets.DownloadableDataset":{_configure_dataset:[10,2,1,""],data_root:[10,3,1,""],download:[10,2,1,""],download_urls:[10,3,1,""]},"emloop.datasets.StreamWrapper":{__enter__:[10,2,1,""],__exit__:[10,2,1,""],__init__:[10,2,1,""],__iter__:[10,2,1,""],__next__:[10,2,1,""],_dequeue_batch:[10,2,1,""],_enqueue_batches:[10,2,1,""],_epoch_limit_reached:[10,2,1,""],_get_stream:[10,2,1,""],_next_batch:[10,2,1,""],_start_thread:[10,2,1,""],_stop_thread:[10,2,1,""],allow_buffering:[10,3,1,""],empty:[10,2,1,""],name:[10,3,1,""]},"emloop.hooks":{AbstractHook:[11,1,1,""],AccumulateVariables:[11,1,1,""],Benchmark:[11,1,1,""],Check:[11,1,1,""],ClassificationMetrics:[11,1,1,""],ComputeStats:[11,1,1,""],EveryNEpoch:[11,1,1,""],Flatten:[11,1,1,""],LogDir:[11,1,1,""],LogProfile:[11,1,1,""],LogVariables:[11,1,1,""],LogitsToCsv:[11,1,1,""],OnPlateau:[11,1,1,""],PlotLines:[11,1,1,""],SaveBest:[11,1,1,""],SaveConfusionMatrix:[11,1,1,""],SaveEvery:[11,1,1,""],SaveFile:[11,1,1,""],SaveLatest:[11,1,1,""],SequenceToCsv:[11,1,1,""],ShowProgress:[11,1,1,""],StopAfter:[11,1,1,""],StopOnNaN:[11,1,1,""],StopOnPlateau:[11,1,1,""],TrainingTerminated:[11,7,1,""],TrainingTrace:[11,1,1,""],WriteCSV:[11,1,1,""]},"emloop.hooks.AbstractHook":{__init__:[11,2,1,""],after_batch:[11,2,1,""],after_epoch:[11,2,1,""],after_epoch_profile:[11,2,1,""],after_training:[11,2,1,""],before_training:[11,2,1,""],register_mainloop:[11,2,1,""]},"emloop.hooks.AccumulateVariables":{__init__:[11,2,1,""],_reset_accumulator:[11,2,1,""],after_batch:[11,2,1,""],after_epoch:[11,2,1,""]},"emloop.hooks.Benchmark":{__init__:[11,2,1,""],after_epoch_profile:[11,2,1,""]},"emloop.hooks.Check":{__init__:[11,2,1,""],after_epoch:[11,2,1,""]},"emloop.hooks.ClassificationMetrics":{__init__:[11,2,1,""],_get_metrics:[11,2,1,""],_save_metrics:[11,2,1,""],after_epoch:[11,2,1,""]},"emloop.hooks.ComputeStats":{EXTRA_AGGREGATIONS:[11,3,1,""],__init__:[11,2,1,""],_compute_aggregation:[11,6,1,""],_raise_check_aggregation:[11,6,1,""],_save_stats:[11,2,1,""],after_epoch:[11,2,1,""]},"emloop.hooks.EveryNEpoch":{__init__:[11,2,1,""],_after_n_epoch:[11,2,1,""],after_epoch:[11,2,1,""]},"emloop.hooks.Flatten":{__init__:[11,2,1,""],after_batch:[11,2,1,""]},"emloop.hooks.LogDir":{__init__:[11,2,1,""],after_epoch:[11,2,1,""],after_training:[11,2,1,""],before_training:[11,2,1,""]},"emloop.hooks.LogProfile":{after_epoch_profile:[11,2,1,""]},"emloop.hooks.LogVariables":{UNKNOWN_TYPE_ACTIONS:[11,3,1,""],__init__:[11,2,1,""],_log_variables:[11,2,1,""],after_epoch:[11,2,1,""]},"emloop.hooks.LogitsToCsv":{__init__:[11,2,1,""],after_batch:[11,2,1,""],after_epoch:[11,2,1,""]},"emloop.hooks.OnPlateau":{OBJECTIVES:[11,3,1,""],_AGGREGATION:[11,3,1,""],__init__:[11,2,1,""],_on_plateau_action:[11,2,1,""],after_epoch:[11,2,1,""]},"emloop.hooks.PlotLines":{__init__:[11,2,1,""],_reset:[11,2,1,""],after_batch:[11,2,1,""],after_epoch:[11,2,1,""],figure_suffix:[11,3,1,""],plot_figure:[11,2,1,""]},"emloop.hooks.SaveBest":{OBJECTIVES:[11,3,1,""],__init__:[11,2,1,""],_get_value:[11,2,1,""],_is_value_better:[11,2,1,""],after_epoch:[11,2,1,""]},"emloop.hooks.SaveConfusionMatrix":{FIGURE_ACTIONS:[11,3,1,""],__init__:[11,2,1,""],after_epoch:[11,2,1,""]},"emloop.hooks.SaveEvery":{SAVE_FAILURE_ACTIONS:[11,3,1,""],__init__:[11,2,1,""],_after_n_epoch:[11,2,1,""],save_model:[11,6,1,""]},"emloop.hooks.SaveFile":{__init__:[11,2,1,""],before_training:[11,2,1,""]},"emloop.hooks.SaveLatest":{__init__:[11,2,1,""],after_epoch:[11,2,1,""]},"emloop.hooks.SequenceToCsv":{__init__:[11,2,1,""],after_batch:[11,2,1,""],after_epoch:[11,2,1,""]},"emloop.hooks.ShowProgress":{__init__:[11,2,1,""],after_batch:[11,2,1,""],after_epoch:[11,2,1,""]},"emloop.hooks.StopAfter":{__init__:[11,2,1,""],_check_train_time:[11,2,1,""],after_batch:[11,2,1,""],after_epoch:[11,2,1,""],before_training:[11,2,1,""]},"emloop.hooks.StopOnNaN":{UNKNOWN_TYPE_ACTIONS:[11,3,1,""],__init__:[11,2,1,""],_check_nan:[11,2,1,""],_is_nan:[11,2,1,""],after_batch:[11,2,1,""],after_epoch:[11,2,1,""]},"emloop.hooks.StopOnPlateau":{_on_plateau_action:[11,2,1,""]},"emloop.hooks.TrainingTrace":{__init__:[11,2,1,""],after_epoch:[11,2,1,""],after_training:[11,2,1,""],before_training:[11,2,1,""]},"emloop.hooks.WriteCSV":{MISSING_VARIABLE_ACTIONS:[11,3,1,""],UNKNOWN_TYPE_ACTIONS:[11,3,1,""],__init__:[11,2,1,""],_write_header:[11,2,1,""],_write_row:[11,2,1,""],after_epoch:[11,2,1,""]},"emloop.models":{AbstractModel:[12,1,1,""],Ensemble:[12,1,1,""],Sequence:[12,1,1,""]},"emloop.models.AbstractModel":{__init__:[12,2,1,""],input_names:[12,3,1,""],output_names:[12,3,1,""],run:[12,2,1,""],save:[12,2,1,""]},"emloop.models.Ensemble":{AGGREGATION_METHODS:[12,3,1,""],__init__:[12,2,1,""],_load_models:[12,2,1,""],input_names:[12,3,1,""],output_names:[12,3,1,""],run:[12,2,1,""],save:[12,2,1,""]},"emloop.models.Sequence":{__init__:[12,2,1,""],_load_models:[12,2,1,""],input_names:[12,3,1,""],output_names:[12,3,1,""],run:[12,2,1,""],save:[12,2,1,""]},emloop:{AbstractDataset:[8,1,1,""],AbstractHook:[8,1,1,""],AbstractModel:[8,1,1,""],BaseDataset:[8,1,1,""],Batch:[8,3,1,""],DownloadableDataset:[8,1,1,""],EpochData:[8,3,1,""],MainLoop:[8,1,1,""],Stream:[8,3,1,""],TimeProfile:[8,3,1,""],constants:[9,0,0,"-"],create_dataset:[8,5,1,""],create_hooks:[8,5,1,""],create_main_loop:[8,5,1,""],create_model:[8,5,1,""],create_output_dir:[8,5,1,""],datasets:[10,0,0,"-"],hooks:[11,0,0,"-"],models:[12,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","data","Python data"],"5":["py","function","Python function"],"6":["py","staticmethod","Python static method"],"7":["py","exception","Python exception"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:data","5":"py:function","6":"py:staticmethod","7":"py:exception"},terms:{"03d":9,"0th":[0,8],"10th":11,"12s":9,"1st":[2,10,11],"2nd":[2,10],"abstract":[8,11,12],"boolean":[4,5],"case":[0,1,2,5,8,10,11,16],"class":[0,1,2,5,16],"default":[4,7,8,9,10,11],"final":[1,4,14,16],"float":[10,11],"function":[1,4,6,10,11,16],"import":[1,2,5,6,9,12,16],"int":[0,1,2,8,10,11,16],"long":[7,11],"new":[0,1,2,5,8,10,11,12,16],"return":[1,4,5,8,10,11,12,16],"short":[2,11],"static":11,"super":2,"true":[0,1,5,8,10,11,12],"try":16,"var":[0,12],"while":[4,5],For:[0,1,2,5,11,16],One:[4,11],That:1,The:[0,2,4,5,6,7,8,9,10,11,12,14,15,16],Then:[8,10,16],There:4,These:[0,1,5],With:[0,1,2,7,16],__enter__:[8,10],__exit__:[8,10],__init__:[0,2,8,10,11,12],__iter__:10,__next__:10,_accumul:11,_after_n_epoch:11,_aggreg:11,_batch_count:11,_check_nan:11,_check_sourc:8,_check_train_tim:11,_compute_aggreg:11,_configure_dataset:[1,8,10,16],_create_model:16,_current_epoch_id:11,_dataset:[11,16],_dequeue_batch:10,_enqueue_batch:10,_epoch:11,_epoch_impl:8,_epoch_limit:2,_epoch_limit_reach:10,_get_metr:11,_get_stream:10,_get_valu:11,_is_nan:11,_is_value_bett:11,_iter:11,_load_model:12,_log_vari:11,_minut:11,_model:12,_next_batch:10,_on_missing_vari:11,_on_plateau_act:11,_on_unknown_typ:11,_on_unused_sourc:8,_raise_check_aggreg:11,_reset:11,_reset_accumul:11,_run_epoch:8,_save_metr:11,_save_stat:11,_start_thread:10,_stop_thread:10,_stream:[1,8,10,11,16],_test_i:16,_test_x:16,_train_i:16,_train_x:16,_var_prefix:11,_variabl:11,_write_head:11,_write_row:11,abil:1,abl:[5,11,16],about:[5,11],abov:[2,11,16],absolut:11,abstract_hook:2,abstractdataset:[0,1,8,10,11,12,16],abstracthook:[0,2,8,11],abstractmodel:[0,4,5,8,10,11,12],accept:[2,5,16],access:7,accord:[8,12,16],accumul:[2,11,12],accumulate_vari:11,accumulatevari:[2,11],accur:12,accuraci:[0,11,16],act:2,action:[0,2,8,11,16],actual:[0,1,2,7],adamoptim:16,add:16,adding:16,addit:[0,2,8,10,12,14,16],addition:[5,8,10,11],advanc:[0,10,15,16],after:[2,4,8,10,11,16],after_batch:[8,11],after_batch_hooks_train:11,after_epoch:[8,11],after_epoch_hook:11,after_epoch_profil:[2,8,11],after_train:[2,8,11],again:[0,16],aggreag:11,aggreg:[11,12],aggregation_method:12,agnost:16,alia:8,alik:5,all:[0,1,2,4,5,6,7,8,10,11,12,16],allow:[1,2,8,10,11,16],allow_buff:10,along:[1,4,16],alongsid:1,alreadi:[2,8,10,11,16],also:[0,2,4,5,6,8,12,16],altern:[4,8,10],although:[5,7],altogeth:2,alwai:[4,10],among:[8,11],amount:11,analog:1,anchor:0,ani:[0,1,2,4,5,6,8,10,11,12,16],anim:[0,1,5],annoi:1,annot:11,anoth:0,answer:16,anyth:16,anywai:11,api:[3,7,8,10,11,15,16],append:[8,12],appli:[11,12],applic:[7,12],approach:0,arbitrari:[1,7,11,16],arbitrarili:0,area:11,aren:0,arg1:2,arg2:2,arg:[2,8,10,12],argument:[0,1,2,4,5,8,11,16],artifact:16,asctim:9,assembl:12,assertionerror:[0,8,11,12],attempt:5,attribut:[0,12,16],attributeerror:8,augment:[0,1],automat:[0,1,5,14,16],avail:[0,2,7,8,10,11],averag:11,avoid:[11,16],axi:[11,16],back:[0,16],backend:[4,5,7,8,12,14],bar:11,base:[4,5,8,10,11,12],base_nam:8,basedataset:[8,10,11,16],basemodel:[5,16],basic:[7,16],batch1:6,batch2:6,batch:[0,1,2,4,5,8,9,10,11,12,16],batch_count:11,batch_data:[2,8,11],batch_siz:[0,1,11,16],batchdata:1,beauti:1,becom:7,been:[2,4],befor:[2,4,6,8,9,11,12,14,16],before_train:[2,8,11],begin:16,behav:6,behavior:[2,5],being:[0,2],below:[5,7,14],benchmark:11,besid:6,best:[1,2,11,16],better:11,between:[0,2,8,11],bia:16,big:[1,16],binari:11,bit:16,blank:11,block:[10,16],blue:11,blur:[0,1],blur_prob:[0,1],bool:[0,8,10,11,12,16],both:[5,11],branch:14,buffer:[0,8,9,10,12],buffer_s:10,build:[0,14,16],built:2,bundl:1,burdensom:7,caffe2:16,calcul:16,call:[1,2,4,5,8,11,12,16],call_native_backend:10,callabl:10,can:[0,1,2,4,5,6,7,10,11,12,14,16],cannot:5,caption:11,care:11,carefulli:2,cast:16,cat:1,catchsigint:2,caughtinterrupt:8,certain:[11,16],chang:[0,16],check:[1,8,10,11,15],checkpoint:[5,8],checksum:1,child:[8,11],childprocesserror:10,class_nam:11,class_with_index_on:11,class_with_index_thre:11,class_with_index_zero:11,classes_nam:11,classes_names_method_nam:11,classif:11,classifi:[0,1],classification_:11,classification_metr:11,classificationinfohook:0,classificationmetr:11,clear:0,cli:[6,8,10],clone:14,closer:2,cmap:11,cntk:16,code:[5,7,14,15],collect:11,color:11,colorbar:11,colormaps_refer:11,column:11,com:[14,15],combin:0,come:[7,11],comfort:0,command:[1,4,5,7,8,10,14,16],comment:[0,1],common:0,compat:1,complet:[1,7,16],complex:1,compli:1,compon:[0,1,3,5,7,16],comput:[0,2,5,8,11,16],compute_stat:11,computestat:[0,2,11,16],concept:[1,8,10],condit:11,config:[0,1,2,4,6,7,8,9,10,11,12,16],config_fil:7,config_path:7,config_str:[8,10],configur:[1,3,5,6,8,9,10,11,12],conflict:11,confus:11,consequ:[4,16],consid:[5,11],consist:16,constant:[8,11,13,16],construct:[0,1,5,8,10,11,16],constructor:[0,1,2,4,5,7,8,10,11,12,16],consum:[1,10],contain:[0,1,2,5,8,9,10,11,16],content:10,context:6,continu:[4,5],contrib:16,control:[11,16],conveni:16,convent:[8,11],convert:[8,16],core:[4,8],correct:16,correctli:11,correspond:[4,5,8,10,11,16],could:1,count:11,counter:11,cover:[8,12],cpu:10,creat:[0,1,2,4,5,8,10,11,12,16],create_dataset:[6,8],create_hook:[6,8],create_main_loop:[6,8],create_model:[6,8],create_output_dir:[6,8],creation:[2,16],criteria:11,crucial:0,csv:[2,11,16],ctrl:2,current:[8,11,12,16],custom:[0,8],cut:[0,8,10],data:[0,2,4,7,8,10,11,16],data_root:[0,8,10],databas:[0,1,16],dataset:[2,3,5,6,8,11,12,13],date:9,debug:[7,16],decid:2,declar:6,decod:[1,8,10,16],deep:0,def:[1,2,10,16],default_model_nam:8,default_valu:11,defaultdict:11,defin:[0,1,5,7,8,10,11,16],delet:[5,7],delimit:11,demand:1,demonstr:[5,6,16],denot:16,dens:16,depend:[4,14],deriv:2,describ:[1,4,8,11,16],descript:16,design:7,desir:[11,16],detail:[2,3,4,16],detect:11,determin:[5,11],dev:14,develop:0,dict:[1,2,8,11,12],dictionari:[0,1,5,6,16],differ:[0,1,5,8,11,12,16],dig:[2,15],dim:16,dimens:16,dir:[7,8,10,11,12],direct:10,directli:[0,8,10,11,14],directori:[0,2,5,6,7,8,9,10,11,12,16],disabl:10,displai:11,distinct:0,distinguish:11,dive:[3,14],divid:[0,13],document:[0,16],doe:[0,8,11,16],doesn:1,dog:1,don:[0,16],done:[0,5,7,8,16],dot:15,doubl:11,download:[1,8,10],download_filenam:[8,10],download_url:[8,10],downloadabledataset:[8,10],drive:1,dtype:[8,10,16],dump:[5,8,9,11],durat:9,dure:[2,4,8,10,11],each:[0,2,4,5,8,11,16],eager_load:12,eas:1,easi:16,easier:16,easili:[0,1,16],either:[8,11,15],el_buffer_sleep:9,el_config_fil:9,el_default_log_dir:9,el_default_train_stream:9,el_full_date_format:9,el_hooks_modul:9,el_log_date_format:9,el_log_fil:9,el_log_format:9,el_na_str:9,el_predict_stream:9,el_trace_fil:9,els:[6,16],eltf:16,embed:0,emloop:[0,1,2,3,4,5,6,13,14],emloop_scikit:0,emloop_tensorflow:[0,2,5,8,16],emploi:16,empti:[0,2,8,10,11],empty_act:[0,8],enabl:[0,1],encod:[0,1,8,16],encourag:[6,16],end:[4,8,10,11,16],enough:16,enqueu:10,ensembl:12,enter:[2,8,11],entri:[0,2,5,8,10,11],environ:[5,10,16],environemt:5,epoch:[0,2,4,6,7,8,10,11,16],epoch_data:[2,8,11],epoch_id:[2,5,8,11],epoch_limit:2,epoch_profil:2,epoch_s:10,epochdata:8,epochstopperhook:2,equal:[11,16],error:[0,8,10,11,16],essenti:1,estim:16,eta:11,etc:[0,1,4,5,7,10,11,16],eval:[0,1,5,8,16],eval_batch_:11,eval_batch_train:11,eval_stream:8,evalu:[1,5,6,7,8,16],even:[2,4],event:[4,8,10,11,16],everi:[1,2,5,6,8,10,11,16],every_n_epoch:11,everynepoch:11,everyth:[2,6,16],exactli:[8,11],exampl:[0,1,2,5,6,8,10,11,16],example_count:11,exc_typ:8,exc_valu:8,exce:11,exceed:11,except:8,execut:[4,11,14,16],exist:[5,8,12,16],expect:[0,2,5,11,16],experi:[0,1,15,16],experiment:6,explain:[7,16],expos:[8,12],extend:[11,16],extens:11,extra:[1,4,8,10,11],extra_aggreg:11,extra_stream:[0,1,4,8,16],extract:[8,10],extrem:[1,16],f1_averag:11,f1s:11,facilit:12,fact:[1,2],fail:11,failur:11,fals:[0,5,7,8,10,11,12],far:11,favorit:16,featur:[0,10,16],fed:[1,4,12],feed:[8,11,12],feel:16,fetch:[1,11,16],few:[0,16],field:1,figsiz:11,figur:[1,11],figure_act:11,figure_suffix:11,file:[0,1,2,5,6,7,8,9,10,11,12,14,16],filenam:9,filesystem:5,find:[1,11,16],fine:4,finish:[2,7,8,11,16],first:[0,1,2,4,5,11,16],five:6,fix:[8,10],fixed_batch_s:[0,8],fixed_epoch_s:[0,8],flag:12,flatten:11,float32:16,flow:[8,11],focu:[7,15],focus:16,follow:[0,1,2,3,4,5,6,7,8,11,13,14,16],form:[0,11,16],format:[1,2,7,9,11],forward:[0,8,12],found:[4,8,11,16],four:[7,16],fraction:16,framework:[2,15,16],free:16,from:[0,1,2,4,5,6,7,8,10,11,12,14,16],fscore:11,full:[9,11,16],fulli:[0,1,10,11],fundament:16,further:[0,1],furthermor:0,gener:[0,8,9,11,16],get:[2,8,10,11],get_stream:8,gil:10,git:14,github:[0,14,15,16],give:1,given:[0,1,2,5,7,8,10,11,12,16],goal:7,going:16,gold_vari:0,good:[2,10,16],graph:16,greater:11,greater_equ:16,green:11,ground:[0,11],gt_variabl:11,handl:[2,11],happen:4,has:[1,2,4,5,8,10,11,14,16],have:[0,2,7,8,10,11,15,16],header:11,height:0,henc:[0,2],here:[4,5,11],hidden:[0,1,16],hold:10,hook:[3,5,6,8,9,13],hour:11,house_s:6,how:[2,6,11,16],howev:[0,1,2,5,6,16],html:11,http:[11,14],id_sourc:11,id_vari:11,ids:[0,11],idx:11,ignor:[1,8,11],imag:[0,1,2,5,11,12],imagin:[1,2],img1:1,img2:1,img3:1,img4:1,implement:[1,2,3,4,5,8,10,12,16],imposs:0,improv:11,includ:[0,7,8,11,16],incorrect_config_act:[0,8],increas:[7,11],index:11,indic:4,individu:[0,11,16],infer:[0,5,8,11,12],inference_logging_hook:0,inferencelogginghook:0,infin:11,info:[2,11,16],inform:[0,1,3,4,5,7,8,12,16],inherit:[1,8,10,11],initi:[1,2,6,8,11],input:[0,5,6,8,11,12,16],input_nam:[5,8,12],insid:2,instal:[7,16],instanc:[0,1,2,5],instanti:5,instead:[0,1,7,11,16],instruct:[2,10],instrument:[2,7,16],intact:4,integr:16,intend:[7,8,10],interfac:[0,5,6,8,11,12,16],internet:1,interoper:1,interrupt:[2,4,8,11],invalid:11,invoc:[8,11],invok:[0,1,2,7,16],ioerror:11,issu:15,item:[4,16],iter:[0,1,4,6,8,10,11,12,16],iterait:[14,15],its:[0,1,2,4,5,6,8,10,11,16],itself:[1,4,5,11],json:[0,16],just:[0,5,7,11,16],keep:[1,2,7,10],kei:[1,2,5,16],kera:16,keyerror:11,know:[0,1,5,15,16],kwarg:[0,1,2,8,10,11,12,16],label:[0,1,2,11],labels_nam:11,languag:[0,1],larg:1,last:[2,11],later:[11,16],latest:11,layer:16,learn:[0,1,2,5,8,11,12,15,16],learning_r:[0,16],least:[7,11],leav:[1,4,8,11],len:16,length:[8,16],let:[2,15,16],level:[1,7,11],levelnam:9,lifecycl:[8,11],lightweight:[15,16],like:[0,2,16],limit:10,line:[0,7,11],link:0,list:[0,1,2,5,6,7,8,10,11,12,14,16],load:[0,1,5,8,12,16],load_training_batch:1,load_yaml:6,log:[0,1,2,5,6,7,8,9,10,11,16],log_dir:[8,11,12],log_profil:[0,11],log_vari:11,logdir:11,logger:8,logit:11,logits_to_csv:11,logitstocsv:11,logprofil:[0,2,8,11],logvari:[2,8,11,16],long_term:11,look:[1,2,5,16],loop:[2,3,5,6,8,11],loss:[0,2,5,11,16],lot:16,lower:11,luckili:2,machin:[0,1,5,8,12,15,16],macro:11,mai:[0,1,2,4,5,8,10,11,16],mail:15,main:[1,2,3,6,8,10,11],main_loop:[0,1,2,4,6,8,11,16],mainloop:[0,5,6,8,10,11,12,16],maintain:11,major:[2,14,16],major_vot:12,majority_dataset:16,majority_net:16,majoritydataset:16,majorityexampl:16,majorityexample_:16,majoritynet:16,make:[1,6,8,10,16],manag:[1,2,5,6,8,10,12,16],mandatori:0,mani:7,map:[8,10,11,12,16],mask:11,mask_nam:11,match:[0,16],matplotlib:11,matrix:11,matter:[1,2,4],max:11,max_epoch:11,maximum:11,mayb:[8,10,12],mean:[0,2,8,10,11,12,16],meant:[11,16],measur:[0,11],median:11,memori:5,messag:9,met:11,method:[0,2,4,5,6,7,8,10,11,12,16],metric:[0,11],micro:11,might:[0,1,4,8,10,14,16],min:11,mini:1,minim:[11,16],minimum:11,minut:[11,15],misc:8,miss:[8,11],missing_variable_act:11,mit:15,mlp:16,model:[1,2,3,6,7,8,10,11,13],model_nam:11,model_path:[7,12],models_path:12,models_root:12,modifi:[2,16],modul:[0,2,8,9,10,11,16],modular:[0,15,16],moment:[1,8,11,16],monitor:11,more:[0,1,2,3,4,7,11,12,16],most:[0,1,5,8,11,16],motiv:0,msec:9,multipl:[4,12],must:[1,6,8,10,11,14,16],my_data:0,my_dataset:[0,1],my_hook:0,my_model:0,my_project:2,my_stream:0,mydataset:[0,1],myensembl:12,myhook:[0,2],mymodel:0,mypipelin:12,n_epoch:[8,11],n_exampl:16,n_hidden_neuron:0,name:[0,1,2,5,8,9,10,11,12,14,16],name_suffix:[5,8,11,12],nan:11,nancount:11,nanfract:11,nanmean:11,natur:[0,8,11,16],necessari:[1,16],need:[0,1,2,5,6,11,16],neither:12,nest:11,net:16,network:[0,16],neural:0,neuron:0,nevertheless:14,new_valu:11,newli:16,next:[1,9,10,16],nightli:14,node:16,non:[0,10],none:[0,2,8,10,11,12,16],nonstandard:6,normal:11,note:[0,2,5,6,16],noth:[2,11],notimplementederror:[8,10,12],now:[1,2,16],npr:16,num:1,num_batch:11,num_class:11,num_classes_method_nam:11,number:[0,2,4,8,10,11,16],numpi:[11,16],object:[0,1,2,5,6,8,10,11,12],obligatori:[8,10],observ:[2,11,16],obtain:[0,8,10],offici:[11,15,16],often:[1,5,11],omnipres:0,on_empty_batch:[0,8],on_empty_stream:[0,8],on_failur:11,on_incorrect_config:[0,8],on_missing_vari:11,on_plateau:11,on_save_failur:11,on_unknown_typ:11,on_unkown_typ:11,on_unused_sourc:[0,8],onc:[0,2,5,8,11,16],one:[0,1,4,5,8,10,11,12,16],ones:[2,16],onli:[0,1,2,4,5,7,8,10,11,12,16],onplateau:11,oper:[1,2,16],opportun:[8,11],optim:16,option:[0,7,8,10,11,12,14,16],order:[1,2,5,8,10,11,12,14],org:11,origin:[0,5,12],other:[1,11,16],otherwis:[8,11,12],our:[0,1,2,5,15,16],out:15,out_format:11,output:[0,2,4,5,6,7,8,9,10,11,12,16],output_dir:[2,8,10,11],output_fil:11,output_nam:[5,8,12],output_root:8,outsid:2,over:[2,16],overal:0,overlap:16,overrid:[0,2,8,10,11],overridden:[0,8,10,11],own:[2,5,11],packag:[13,16],pad:11,pad_mask_vari:11,paramet:[0,1,5,8,10,11,12,16],parametr:0,pars:[1,8,10],part:[0,5,16],particular:[1,11,16],pass:[0,1,2,4,5,6,7,8,10,11,12,16],path:[0,5,7,8,10,11,12],peek:2,perfectli:4,perform:[1,4,11,16],performac:16,period:11,persist:5,phase:[0,4],picture_id:11,pip:14,pipelin:1,place:[2,10,11],placehold:16,plateau:11,pleas:[15,16],plot:[1,11],plot_figur:11,plot_histogram:1,plot_lin:11,plotlin:11,plugin:14,png:11,point:[1,2,5],portion:1,posibl:11,posit:5,possibl:[6,8,10,11,12],pow:16,practic:1,preced:12,precis:[8,11,12,16],precision_recall_fscore_support:11,predict:[0,2,7,9,11,12,16],predict_stream:0,predicted_anim:5,predicted_vari:[0,11],predictions_nam:11,predit:2,prefix:11,prepar:10,prepare_stream:8,preprocess_batch_in_python:10,prescrib:[8,10],present:[11,16],pretti:9,previou:6,price:6,principl:16,print:[1,7,9,11],print_statist:1,prior:4,probabl:[0,1],problem:1,proce:16,procedur:10,process:[0,2,4,5,6,8,9,11,12,16],produc:[0,10,11,16],product:[5,16],profil:[2,4,8,10,11],programmat:7,progress:[11,16],project:[12,15,16],proper:7,properli:[6,10,11,14,16],properti:[5,8,10,11],provid:[0,1,2,5,6,8,11,12,16],pseudocod:[4,5],publish:5,purpos:[1,14,16],put:7,python:[1,2,3,8,10,12,14,16],qualifi:[0,1,11],queri:[1,4],queue:10,quit:[1,2,16],rabbit:1,rais:[0,2,8,10,11,12],random:[0,1,16],random_integ:16,randomli:16,rang:[1,5,6,16],rapid:[6,15,16],rate:0,ratio:16,raw:10,reach:[10,11],read:[11,16],read_data_train:11,readi:16,real:16,reason:[0,1],recal:11,receiv:[4,8],recogn:[8,11],recognit:5,recommend:[1,14,16],record:[10,11],recurs:[7,11],red:11,refer:[0,11,15,16],referenc:0,regardless:[1,2,8,11],regist:[0,1,4,8,16],register_mainloop:[8,11],regular:16,rel:11,relat:[1,16],releas:10,releasedsemaphor:10,remain:0,remov:[0,8],renam:16,report:[8,10],repositori:[0,14,16],repres:[2,11,16],requir:[0,1,6,8,10,11,16],required_min_valu:11,reset:[10,11],reshap:16,resolut:0,resourc:10,respect:[1,2,8,11,16],respons:[4,16],restor:[4,8],restore_fallback:5,restore_from:[5,7,8,12],result:[0,2,8,11,12],resum:[5,16],resus:0,retriev:[6,11],reusabl:16,right:4,risen:10,root:[0,8,10,12],root_dir:11,rotat:[0,1],roughli:2,routin:[8,10],row:11,run:[4,6,8,10,11,12,14,16],run_evalu:8,run_train:8,same:11,sampl:11,save:[0,2,5,8,10,11,12,16],save_cm:11,save_failure_act:11,save_fil:11,save_model:11,savebest:[2,8,11,16],saveconfusionmatrix:11,saveeveri:11,savefil:11,savelatest:11,scalar:11,scenario:8,scikit:11,scratch:8,script:1,seamlessli:1,search:11,second:[2,5,16],section:[0,1,3,4,8,11,12,15,16],see:[0,1,2,3,5,11,14,15,16],select:[4,11],self:[1,2,8,10,11,12,16],separ:[0,1,16],sequenc:[8,10,11,12],sequence_to_csv:11,sequencetocsv:11,sequenti:12,set:[0,1,8,10,11,12],setup:14,sever:[1,13],shall:[5,8,16],shape:[8,10,16],share:[2,7,8,11],short_term:11,should:[1,4,5,7,8,10,11,12,16],show:[6,11],show_progress:11,showprogress:11,signal:10,similar:[1,16],similarli:16,simpl:[0,1,2,4,12,16],simplest:14,simpli:[0,16],simultan:10,singl:[4,5,6,7,8,10,11,16],situat:0,size:[0,1,4,8,10,11],skeleton:1,skip:[0,8],skip_zeroth_epoch:[0,8],sklearn:11,sleep:9,slightli:16,small:1,smaller:11,snippet:5,solv:16,some:[0,1,2,8,10,11,16],somebodi:5,sometim:[0,1,11],soon:11,sourc:[0,1,2,5,8,10,11,12,14,15,16],source_nam:11,spare:10,special:[0,2],specif:[4,5,7,8,11,12],specifi:[0,1,2,4,5,7,8,10,11,12,16],spent:11,split:[1,16],squar:16,stabl:11,standard:[0,2,6,9,11],start:[2,4,7,8,9,10,11],stat:11,state:10,statist:[1,5,11],stderr:11,stem:[8,11],step1:12,step2:12,step3:12,step:[4,12,16],stop:[2,7,10,11,16],stop_aft:11,stop_ev:10,stop_on_inf:11,stop_on_nan:11,stop_on_plateau:11,stopaft:[2,8,11,16],stopiter:10,stoponnan:11,stoponplateau:11,store:[2,5,7,8,11,12,16],str:[0,5,8,10,11,12],straightforward:6,stream:[0,2,4,6,7,8,9,10,11,12,16],stream_fn:10,stream_info:[8,10],stream_list:8,stream_nam:[1,2,5,7,8,10,11],streamwrapp:[8,10,12],string:[0,1,6,8,9,10,11,16],structur:[0,2],stuff:10,sub:[0,7,12],subclass:5,subdir:7,subdirectori:7,subsequ:2,subset:11,success:[2,8,11],successfulli:[5,8,11],suffix:[8,11,12,16],suggest:1,suit:[0,14],sum:16,summar:11,superset:0,supplement:11,suppli:2,support:[0,11,12],system:2,t_co:[8,10],take:[0,2,8,10,11],taken:[8,11,12,16],target:[0,16],techniqu:[1,6],tell:16,ten:11,tensorboard:2,tensorboardhook:0,tensorflow:[5,14,16],tenth:6,termin:[0,2,4,8,10,11],test:[0,1,2,4,8,11,14,16],test_batch:5,test_stream:[1,5,16],than:[2,11],thank:[8,11],thei:[0,1,16],them:[1,2,8,10,11,12,16],therefor:[0,4,16],thi:[0,1,2,3,4,5,7,8,10,11,12,14,16],thing:[6,16],those:[4,16],though:1,thread:10,three:11,threshold:11,through:[1,4,7,8,11,16],thu:16,time:[1,8,10,11],timeprofil:8,tmp:11,togeth:[0,1,7,12],token:0,total:[11,16],total_batch_count:11,trace:[9,11],traceback:8,track:2,tracker:15,train:[0,1,2,5,6,8,9,10,11,12,16],train_batch:5,train_stream:[1,4,5,6,8,10,11,16],train_stream_nam:[0,8,11],trainabl:[8,12],trainig:2,training_epochs_don:8,training_trac:11,trainingtermin:[2,8,11],trainingtrac:11,translat:[2,11],trigger:[2,4,8,11,16],truli:1,truth:[0,11],tupl:[1,11],tutori:[2,5,15],two:[1,2,4,5,8,11],txt:14,type:[0,8,10,11,12],typeerror:11,typic:[1,8,10,12,16],ultim:12,unannot:0,undefin:11,under:[1,2,8,11,12,15],underli:[10,12],understand:16,unexpect:[0,8],unexpectedli:10,union:11,uniqu:11,unit:16,unknown:11,unknown_type_act:11,unnam:8,unsupport:[0,8,11],unsurprisingli:0,until:8,unus:[0,8,12],unused_source_act:[0,8],updat:[1,4,5,8,12],upon:2,url:[8,10],url_root:[8,10],usabl:[15,16],usag:[1,7,10,12],use:[0,1,2,4,6,7,8,10,14,16],used:[0,1,5,7,8,9,10,11,12,16],useful:[1,2,8,12,16],useless:1,user:[6,8,11,14],uses:[0,10],using:[6,7,11,12,14,16],usual:[2,4,5,6,7,8,11,12,16],util:[6,8,10,11,14],valid:[0,1,2,4,6,8,11,16],valid_stream:[1,6],valu:[0,2,4,8,10,11,16],valueerror:[8,11,12],var_prefix:11,variabl:[0,2,5,8,10,11,16],variable_nam:11,variou:[0,1,9,16],vector:16,verb:[8,11],verbos:7,veri:[1,4,16],verif:16,via:[6,8,10,11,15],video_id:11,visual:[0,1,11],wai:[0,7,14],want:16,warn:[0,8,11],wast:1,watch:[2,16],weight:11,well:[0,11,12,16],were:[8,11,12],what:[0,2,4,5,16],when:[0,1,4,5,6,7,8,10,11,12,16],whenc:8,whenev:[6,10],where:[0,2,4,5,8,11,16],wherein:[0,8,11],whether:[1,4,5,8,10,11],which:[0,1,2,4,5,7,8,9,10,11,12,14,16],whichev:11,whole:[0,1,2,4,5,7,16],width:0,wish:[1,5],within:11,without:[1,5,7,10,11],won:1,word:16,work:1,workflow:5,world:16,would:[0,1,2,5,9,10],wrap:16,wrapper:[8,10,12],write:[1,2,11],write_csv:11,writecsv:[2,11],writetensorboard:[2,8],written:[2,8,10,11],wrong:11,xs_flat:11,xxx:11,y_hat:16,yaml:[0,1,6,8,9,10,11,16],yet:[0,5,8,11],yield:[1,16],ymax:11,ymin:11,you:[0,1,2,5,15,16],your:[0,1,2,4,16],zero:16},titles:["Configuration","Dataset","Hooks","Advanced","Main Loop","Model","Python API","CLI Reference","<code class=\"docutils literal notranslate\"><span class=\"pre\">emloop</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">emloop.constants</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">emloop.datasets</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">emloop.hooks</span></code>","<code class=\"docutils literal notranslate\"><span class=\"pre\">emloop.models</span></code>","API Reference","Getting Started","emloop","Tutorial"],titleterms:{"class":[8,10,11,12],"function":8,The:1,Using:16,addit:1,advanc:3,after_batch:2,after_epoch:2,api:[6,13],argument:7,basedataset:1,cli:7,conclus:0,configur:[0,2,16],constant:9,contribut:15,data:1,dataset:[0,1,4,7,10,16],develop:14,emloop:[7,8,9,10,11,12,15,16],eval:7,evalu:[0,4],event:2,except:11,get:14,hook:[0,2,4,11,16],instal:14,integr:4,introduct:16,lazi:1,licens:15,lifecycl:4,loop:[0,4,16],main:[0,4,16],method:1,model:[0,4,5,12,16],name:7,philosophi:1,posit:7,prune:7,python:6,refer:[7,13],regular:2,requir:14,restor:5,resum:7,run:5,start:14,stream:1,submodul:8,support:15,task:16,train:[4,7],tutori:16,variabl:9}})