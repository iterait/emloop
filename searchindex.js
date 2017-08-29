Search.setIndex({docnames:["advanced/config","advanced/dataset","advanced/hook","advanced/index","advanced/main_loop","advanced/model","cli","cxflow/cxflow","cxflow/cxflow.constants","cxflow/cxflow.datasets","cxflow/cxflow.hooks","cxflow/cxflow.models","cxflow/index","getting_started","index","tutorial"],envversion:51,filenames:["advanced/config.rst","advanced/dataset.rst","advanced/hook.rst","advanced/index.rst","advanced/main_loop.rst","advanced/model.rst","cli.rst","cxflow/cxflow.rst","cxflow/cxflow.constants.rst","cxflow/cxflow.datasets.rst","cxflow/cxflow.hooks.rst","cxflow/cxflow.models.rst","cxflow/index.rst","getting_started.rst","index.rst","tutorial.rst"],objects:{"":{cxflow:[7,0,0,"-"]},"cxflow.MainLoop":{PREDICT_STREAM:[7,2,1,""],TRAIN_STREAM:[7,2,1,""],UNUSED_SOURCE_ACTIONS:[7,2,1,""],__init__:[7,3,1,""],__weakref__:[7,2,1,""],_check_sources:[7,3,1,""],_create_epoch_data:[7,3,1,""],_run_epoch:[7,3,1,""],_run_zeroth_epoch:[7,3,1,""],evaluate_stream:[7,3,1,""],get_stream:[7,3,1,""],run_prediction:[7,3,1,""],run_training:[7,3,1,""],train_by_stream:[7,3,1,""]},"cxflow.constants":{CXF_CONFIG_FILE:[8,4,1,""],CXF_FULL_DATE_FORMAT:[8,4,1,""],CXF_HOOKS_MODULE:[8,4,1,""],CXF_LOG_DATE_FORMAT:[8,4,1,""],CXF_LOG_FILE:[8,4,1,""],CXF_LOG_FORMAT:[8,4,1,""]},"cxflow.datasets":{AbstractDataset:[9,1,1,""],BaseDataset:[9,1,1,""]},"cxflow.datasets.AbstractDataset":{__init__:[9,3,1,""],__weakref__:[9,2,1,""]},"cxflow.datasets.BaseDataset":{__init__:[9,3,1,""],_init_with_kwargs:[9,3,1,""]},"cxflow.hooks":{AbstractHook:[10,1,1,""],AccumulateVariables:[10,1,1,""],CatchSigint:[10,1,1,""],Check:[10,1,1,""],ComputeStats:[10,1,1,""],LogProfile:[10,1,1,""],LogVariables:[10,1,1,""],SaveBest:[10,1,1,""],SaveEvery:[10,1,1,""],StopAfter:[10,1,1,""],TrainingTerminated:[10,6,1,""],WriteCSV:[10,1,1,""]},"cxflow.hooks.AbstractHook":{__init__:[10,3,1,""],__weakref__:[10,2,1,""],after_batch:[10,3,1,""],after_epoch:[10,3,1,""],after_epoch_profile:[10,3,1,""],after_training:[10,3,1,""],before_training:[10,3,1,""]},"cxflow.hooks.AccumulateVariables":{__init__:[10,3,1,""],_reset_accumulator:[10,3,1,""],after_batch:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.hooks.CatchSigint":{__init__:[10,3,1,""],_sigint_handler:[10,3,1,""],after_batch:[10,3,1,""],after_training:[10,3,1,""],before_training:[10,3,1,""]},"cxflow.hooks.Check":{__init__:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.hooks.ComputeStats":{AGGREGATIONS:[10,2,1,""],__init__:[10,3,1,""],_compute_aggregation:[10,5,1,""],_save_stats:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.hooks.LogProfile":{after_epoch_profile:[10,3,1,""]},"cxflow.hooks.LogVariables":{UNKNOWN_TYPE_ACTIONS:[10,2,1,""],__init__:[10,3,1,""],_log_variables:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.hooks.SaveBest":{OBJECTIVES:[10,2,1,""],__init__:[10,3,1,""],_get_value:[10,3,1,""],_is_value_better:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.hooks.SaveEvery":{SAVE_FAILURE_ACTIONS:[10,2,1,""],__init__:[10,3,1,""],after_epoch:[10,3,1,""],save_model:[10,5,1,""]},"cxflow.hooks.StopAfter":{__init__:[10,3,1,""],_check_train_time:[10,3,1,""],after_batch:[10,3,1,""],after_epoch:[10,3,1,""],before_training:[10,3,1,""]},"cxflow.hooks.WriteCSV":{MISSING_VARIABLE_ACTIONS:[10,2,1,""],UNKNOWN_TYPE_ACTIONS:[10,2,1,""],__init__:[10,3,1,""],_write_header:[10,3,1,""],_write_row:[10,3,1,""],after_epoch:[10,3,1,""]},"cxflow.models":{AbstractModel:[11,1,1,""]},"cxflow.models.AbstractModel":{__init__:[11,3,1,""],__weakref__:[11,2,1,""],input_names:[11,2,1,""],output_names:[11,2,1,""],restore_fallback:[11,2,1,""],run:[11,3,1,""],save:[11,3,1,""]},cxflow:{MainLoop:[7,1,1,""],constants:[8,0,0,"-"],datasets:[9,0,0,"-"],hooks:[10,0,0,"-"],models:[11,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","data","Python data"],"5":["py","staticmethod","Python static method"],"6":["py","exception","Python exception"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:data","5":"py:staticmethod","6":"py:exception"},terms:{"06d":8,"0th":7,"10th":10,"15s":8,"1st":2,"2nd":2,"800x600":1,"800x600x3":1,"abstract":11,"boolean":[4,5],"case":[0,1,2,5,10,15],"catch":10,"class":[0,1,2,5,15],"default":[0,4,6,10],"final":[0,1,4,13,15],"float":10,"function":[1,4,7,10,15],"import":[0,1,2,5,8,15],"int":[1,2,7,10,15],"new":[0,2,5,9,10,15],"public":3,"return":[4,5,7,9,10,11,15],"short":2,"static":10,"super":2,"switch":10,"true":[0,1,5,7,11],"try":15,"var":0,"while":[0,4,5],For:[0,1,2,3,5,10,15],One:[4,10],That:[1,15],The:[0,2,3,4,5,6,9,10,13,14,15],There:4,These:[0,5],With:[6,14],__init__:[2,7,9,10,11],__minut:10,__weakref__:[7,9,10,11],_accumul:10,_check_sourc:7,_check_train_tim:10,_compute_aggreg:10,_create_epoch_data:7,_create_model:15,_epoch_limit:2,_get_valu:10,_init_with_kwarg:[1,9,15],_is_value_bett:10,_iter:10,_log_vari:10,_on_missing_vari:10,_on_unknown_typ:10,_on_unused_sourc:7,_reset_accumul:10,_run_epoch:7,_run_zeroth_epoch:7,_save_every_n_epoch:10,_save_stat:10,_sigint_handl:10,_stream:[1,9,10,15],_test_i:15,_test_x:15,_train_i:15,_train_x:15,_variabl:10,_write_head:10,_write_row:10,abl:[1,5,15],about:[5,10],abov:[0,2,10,15],abstract_hook:[2,10],abstractdataset:[7,9,11,15],abstracthook:[2,7,10],abstractmodel:[4,5,7,10,11],accept:[1,2,5,15],access:[6,15],accord:[11,15],accumul:[2,10],accumulate_vari:10,accumulatevari:[2,10],accuraci:[0,2,10,15],act:2,action:[7,10,15],actual:[0,1,2,6],adamoptim:15,add:[9,15],adding:15,addit:[0,2,4,7,9,10,11,13,15],addition:[5,9,10],advanc:[0,14,15],advantag:0,after:[0,2,4,10,15],after_batch:[3,7,10],after_batch_hooks_train:10,after_epoch:[7,10],after_epoch_hook:10,after_epoch_profil:[2,7,10],after_train:[2,7,10],afterward:4,again:[0,15],aggreag:10,aggreg:10,algorithm:15,all:[0,1,2,5,6,7,9,10,11,15],allow:[2,6,10,15],almost:0,along:[1,4],alreadi:[0,1,2,3,15],also:[2,4,5,11,15],altern:4,although:[5,6],altogeth:2,among:[3,10],analog:[0,1],anchor:0,ani:[1,2,5,7,9,10,15],anim:[0,1,5],annoi:1,anoth:[0,3],answer:15,anyth:15,anywai:10,api:[3,6,9,10,14,15],append:11,approach:0,approch:0,appropri:0,arbitrari:[1,15],arbitrarili:0,aren:0,arg1:2,arg2:2,arg:2,argument:[0,1,2,4,5,6,10,15],around:15,artifact:15,artifici:1,asctim:8,assum:1,attempt:5,attribut:[0,3,15],attributeerror:7,augment:[0,1],automat:[0,1,5,13,15],avail:[2,6,9,10],avoid:15,axi:15,back:0,backend:[4,5,11,13],base:[4,7,9,10,11],base_dataset:9,baseclass:[5,11],basedataset:[9,15],basemodel:[5,15],basic:[6,15],batch:[0,1,2,3,4,5,7,10,11,15],batch_data:[2,10],batch_siz:[0,1,15],becom:6,been:[2,4],befor:[0,2,4,10,13,15],before_train:[2,7,10],begin:15,behavior:[0,2,5],being:2,bellow:6,below:[3,5,13,15],best:[1,2,10,15],bestsaverhook:[10,15],better:10,between:[2,7],big:[1,15],bit:15,blank:10,block:15,blur:[0,1],blur_prob:[0,1],bool:[7,10,11,15],both:[0,5],branch:13,broken:12,build:[0,4,13,15],built:2,burdensom:6,caffe2:15,calcul:15,call:[2,4,5,7,10,15],can:[0,1,2,3,4,5,6,13,14,15],cannot:5,carefulli:2,cast:15,cat:1,catch_sigint:10,catchsigint:[2,10,15],caught:10,certain:10,chang:[0,15],check:[1,7,10,14],checkpoint:5,checksum:1,child:10,classif:[1,2],classifi:[0,1],classificationinfohook:[0,2],clear:0,cli:1,clone:13,closer:2,cntk:15,code:[5,6,10,13,14],cognexa:[13,14],collect:[1,10],column:10,com:[13,14],combin:0,come:6,comfort:0,command:[1,4,5,6,13,15],comment:[0,1],common:0,complet:[0,1,15],compon:[0,3,5,6,14,15],comput:[0,2,5,10,15],compute_stat:10,computestat:[0,2,10,15],concept:[1,9],condit:10,config:[0,1,2,4,8,9,15],config_str:9,configur:[1,3,5,8,9,10,11],consequ:15,consid:5,consist:[1,3,15],constant:[7,12,15],construct:[0,1,5,10,11,15],constructor:[0,1,2,3,4,5,9,11,15],consum:1,contain:[0,1,2,5,8,9,10,15],continu:5,contrari:[3,15],contrib:15,conveni:0,convent:[1,10,15],convert:15,core:4,correct:15,correspond:[1,4,5,15],could:1,count:10,counter:10,cover:[11,15],creat:[1,2,4,5,7,9,10,15],creation:[2,15],criteria:10,crucial:0,csv:[2,10,15],ctrl:2,current:[0,10,11,15],custom:[0,15],cxf_config_fil:8,cxf_full_date_format:8,cxf_hooks_modul:8,cxf_log_date_format:8,cxf_log_fil:8,cxf_log_format:8,cxflow:[0,1,2,3,4,5,12,13],cxflow_scikit:[0,2],cxflow_tensorflow:[0,2,5,15],cxtf:15,dash:3,data:[0,2,4,7,10,15],data_root:0,databas:[0,1,15],dataset:[2,3,5,7,11,12],date:8,datset:0,debug:[6,15],decid:2,decod:[1,9,15],deep:[0,14],def:[1,2,15],default_valu:10,defaultdict:10,defin:[0,1,5,6,7,9,10,11,14,15],delet:5,delimit:10,demonstr:[1,3,5,15],denot:15,dens:15,dense1:15,dense2:15,depend:[4,13],depict:1,deploi:0,deriv:2,describ:[1,4,10,15],descript:15,design:[6,15],desir:15,detail:[2,3,4,15],detect:10,determin:[5,10],dev:[13,15],develop:[0,1,14],deviat:15,dict:[0,1,2,7,10,11],dictionari:[5,10,15],differ:[0,5],dig:[2,14],dim:15,dimens:15,dir:[9,11],directli:[0,10,13,15],directori:[0,2,5,6,8,9,10,11,15],distinct:0,dive:[3,13],divis:10,document:[0,15],doe:[0,4,7,10,15],doesn:[0,1],dog:1,don:[0,1,15],done:[0,5,15],dot:0,doubl:[7,10],down:12,download:1,drive:1,dtype:15,dump:[5,8],duplic:0,dure:[0,1,2,4,10,15],each:[0,2,3,4,5,10,15],easi:15,easier:15,easili:[0,1,14,15],either:[0,1,10,14],eleg:1,els:15,embed:0,emploi:[0,1,15],empti:[2,7,10],enabl:[0,1],encapsul:1,encod:[0,1,15],encourag:15,end:4,enforc:0,enough:15,enter:[2,10],entri:[0,2,5,9,10],enumer:10,environ:[5,15],environemt:5,epoch:[0,2,4,7,10,15],epoch_data:[2,10],epoch_id:[2,5,10],epoch_limit:2,epoch_profil:2,epochstopperhook:2,equal:[10,15],error:[0,7,10,15],estim:15,etc:[1,4,5,6,10],eval_batch_train:10,evalu:[0,1,4,5,7,15],evaluate_stream:7,evaluet:0,even:[2,4],event:[4,7,10,15],everi:[2,5,9,10,15],everyth:[2,15],exactli:[0,10],exampl:[0,1,2,3,5,10,14,15],exce:10,exceed:10,execut:[1,4,10,13,15],exist:[5,11],exit:10,expect:[0,1,2,5,10,15],experi:[0,1],explain:[6,15],explicit:15,expos:11,extend:[10,15],extra:[1,4],extra_stream:[0,1,4,7,10,15],extrem:[1,15],fact:2,fail:[10,11],failur:10,fallback:11,fals:[5,7,11],far:10,favourit:15,featur:[0,15],fed:[0,1,4],feed:11,feel:15,fetch:[1,15],few:[0,15],figur:[1,3],file:[0,1,2,5,8,10,11,13,15],filesystem:5,fine:[0,4],finish:[2,10,15],first:[0,1,2,4,5,10,15],fixed_batch_s:[0,7],float32:15,flow:10,focu:6,focus:15,follow:[0,1,2,3,4,5,6,12,13,15],form:[0,1,10,15],format:[2,8,10],former:1,forward:11,found:[4,10,15],four:[1,6,15],fraction:15,framework:[2,15],free:15,from:[0,1,2,4,5,9,10,13,15],fscore:10,full:[8,10],fulli:[0,10,11],fundament:15,further:[0,1],furthermor:0,futur:[14,15],gener:[0,1,8,15],get:[2,7,15],get_stream:7,git:13,github:[13,14],given:[0,1,2,5,7,9,10,11,15],goal:6,going:15,gold_vari:[0,2],good:[2,15],grace:10,graph:15,greater_equ:15,ground:0,handl:[1,2],handler:10,happen:[4,15],has:[1,2,4,5,9,10,13],have:[0,2,7,9,10,14,15],header:10,height:[0,1],henc:[0,1,2,15],here:[5,10],hidden:[0,1,15],hidden_activ:15,hook:[3,5,7,8,12],how:[2,10,15],howev:[0,1,2,5],http:[13,14],idea:15,ident:0,ids:0,ignor:[0,1,7,10],imag:[0,1,2,5],imagin:2,img1:1,img2:1,img3:1,img4:1,immedi:[0,10],implement:[1,2,3,4,5,9,11,15],imposs:0,includ:[0,10,15],incorrectli:0,increas:[6,10],index:10,indic:4,individu:[0,15],infer:[1,5,7,10],inference_logging_hook:0,inferencelogginghook:0,info:[2,10],inform:[0,3,4,5,11,15],inherit:[1,9,10],initi:[2,9,10],input:[0,1,5,10,11,15],input_nam:[5,11],insid:2,insight:3,instal:[6,15],instanc:[0,1,2,5],instanti:5,instead:[0,1,6,9,15],instruct:2,instrument:[2,6],integr:15,intend:6,interfac:[0,5,10,11,15],internet:1,interrupt:[2,4,10],introduc:[0,15],invoc:[3,10],invok:[0,1,2,3,15],ioerror:10,issu:14,item:[4,15],iter:[1,4,7,10,11,15],its:[0,1,2,4,5,10,15],itself:[0,4,10],json:[0,15],just:[0,5,6],keep:[1,2],kei:[1,5,15],kera:15,keyerror:10,know:[0,1,5,14,15],kwarg:[0,1,2,9,10,11,15],label:[0,2],languag:0,last:2,later:[10,15],latter:1,layer:15,lazili:1,learn:[0,2,5,11,14,15],learning_r:[0,15],least:10,leav:1,len:15,length:15,less:0,let:[1,2,14,15],level:[0,6,10],levelnam:8,lifecycl:10,like:[2,10,15],line:[0,3,6,10],link:0,list:[0,2,3,5,7,9,10,11,13,14,15],load:[0,1,5,7,15],load_training_batch:1,locat:[1,14],log:[0,2,5,6,8,9,10,15],log_dir:11,log_profil:[0,10],log_vari:10,logprofil:[0,2,10],logvari:[2,10,15],longer:0,look:[1,2,5,15],loop:[2,3,5,7,10],loss:[0,2,5,10,15],lot:15,luckili:2,machin:[0,5,11,15],mai:[0,1,2,5,9,10,15],mail:14,main:[1,2,3,7,10],main_loop:[0,1,2,4,7,15],mainloop:[3,5,7,9,10,11,15],maintain:10,major:[2,13,15],majoritydataset:15,majorityexampl:15,majorityexample_:15,majoritynet:15,make:[9,15],manag:[2,5,14],mandatori:0,map:[1,7,10,11,15],match:0,matter:[2,4],max:10,max_epoch:10,maximum:10,mean:[0,2,10,15],meant:[10,15],measur:[0,10],median:10,memori:5,mention:3,messag:8,met:10,method:[0,2,3,4,5,9,10,11,15],metric:[0,10],might:[0,1,4,13,15],min:10,minim:[10,15],minut:[10,14],miss:[7,10],missing_variable_act:10,mit:14,mix:0,mlp:15,mode:7,model:[1,2,3,6,7,10,12,14],modifi:2,modul:[0,1,2,7,8,9,10,11,15],modular:[0,15],moment:[1,10,15],monitor:10,more:[0,2,3,4,10,15],most:[5,10,15],motiv:0,msec:8,multipl:[3,4],must:[0,9,13],mutual:3,my_data:0,my_dataset:[0,1],my_hook:0,my_model:0,my_modul:0,my_project:2,mydataset:[0,1],myhook:[0,2],mymodel:0,n_epoch:10,n_exampl:15,n_hidden_neuron:0,name:[0,1,2,4,5,7,8,10,11,13,15],name_suffix:[5,10,11],natur:[0,10,15],necessari:[1,15],need:[0,1,2,5,15],nest:[0,10],net:4,network:[0,15],neural:0,neuron:0,never:4,nevertheless:[0,13],new_valu:10,newli:15,next:15,nightli:13,node:15,none:[2,7,9,10,11,15],nonetyp:[7,10,11],note:[0,1,2,5,15],noth:[2,10],notimplementederror:9,now:[0,2,15],npr:15,num:1,number:[0,2,4,10,15],numpi:[10,15],object:[0,1,2,3,5,7,9,10,11],obligatori:9,observ:[0,2,15],obtain:[0,9],offici:[10,14,15],often:[5,10],on_failur:10,on_missing_vari:10,on_save_failur:10,on_unknown_typ:10,on_unused_sourc:[0,7],onc:[0,2,3,5,10,15],one:[1,2,3,4,5,7,9,10,15],ones:[0,2,15],onli:[0,1,2,3,4,5,10,15],oper:[1,2,15],opportun:10,optim:15,option:[0,7,9,10,11,13,15],order:[0,1,2,3,5,9,10,13,15],origin:[5,10],orthogon:3,other:[0,1,10,15],otherwis:11,our:[1,2,5,14,15],out:14,output:[0,2,4,5,6,8,9,10,11,15],output_dir:[2,9,10],output_fil:10,output_nam:[5,10,11],outsid:2,over:[2,15],overal:[0,3],overlap:15,overrid:[2,9],overridden:9,own:[2,5,10],packag:[12,15],param:11,paramet:[0,1,5,7,9,10,11,15],pars:[1,9],part:[0,5,15],pass:[0,2,3,4,5,9,11,15],path:[5,11],peek:2,perfectli:4,perform:[1,4,10,15],performac:15,persist:5,phase:[0,4],pip:13,pipelin:1,place:2,placehold:15,pleas:[14,15],plot:1,plot_histogram:1,plugin:13,point:[0,2,5],polit:10,posibl:10,posit:5,possibl:[0,1,3,7,10,15],pow:15,practic:1,precis:15,predict:[0,1,2,5,7,15],predict_stream:[1,7],predicted_anim:5,predicted_vari:[0,2],predit:2,prescrib:9,present:[10,15],print:[1,10],print_statist:1,probabl:[0,1],proce:15,process:[0,2,3,4,5,10,11,15],produc:[0,10,15],product:[0,5,15],profil:[2,4,10],programmat:6,progress:15,project:[14,15],proper:[1,6],properli:[10,13],properti:5,provid:[0,1,2,3,5,7,10,12,15],pseudocod:[4,5],publish:5,purpos:[1,13,15],put:6,python:[1,2,9,13,15],qualifi:[0,10,11],queri:[1,4],quit:[2,10],rabbit:1,rais:[2,7,9,10],random:[0,1,15],random_integ:15,randomli:15,rang:[1,5,15],rate:0,ratio:15,reach:10,read:10,read_data_train:10,readi:[0,15],real:15,realli:[1,15],reason:[0,1],receiv:7,recogn:10,recognit:5,recommend:[0,1,13,15],rectangl:3,reduc:0,reduce_mean:15,refer:[1,7,9,10,11,14],regardless:[2,10],regist:[0,1,4,10,15],registr:[0,15],regular:[1,15],relat:[1,15],relationship:3,remain:0,remov:7,renam:15,repositori:13,repres:[1,2,3,15],requir:[0,1,9,10,15],required_min_valu:10,reset:10,reshap:15,resolut:0,respect:[2,10],respons:[4,15],rest:0,restor:11,restore_fallback:[5,11],restore_from:[5,11],result:[0,2,10,11],resum:[1,5,15],resus:0,retriev:10,reus:0,reusabl:[14,15],rewrit:0,rgb:1,right:4,rotat:[0,1],roughli:2,row:10,rtype:11,run:[4,7,10,11,13,15],run_predict:7,run_train:7,safe:15,same:0,save:[0,2,5,10,11,15],save_failure_act:10,save_model:10,savebest:[2,10],saveeveri:10,scalar:10,score:2,script:1,second:[0,1,2,5,10,15],section:[0,1,3,4,11,14,15],see:[0,2,3,5,10,13,14,15],select:4,self:[1,2,7,10,15],self_num_sigint:10,separ:[0,1,15],serv:0,set:[0,1,7,10],setup:13,sever:[1,12],shall:[2,5,15],shape:[1,15],share:[2,6,10],should:[1,4,5,6,9,10,11,15],sigint:[10,15],signal:[10,15],signum:10,similar:5,similarli:15,simpl:[0,1,2,4,15],simpler:15,simplest:13,simpli:15,simplic:1,sinc:0,singl:[1,3,4,5,10,15],situat:0,size:[0,1,4,7],skip:[0,7],skip_zeroth_epoch:[0,7],smart:[14,15],snippet:[0,5,15],solid:3,some:[0,1,2,15],somebodi:5,sometim:1,sourc:[0,1,2,5,7,9,10,11,13,14,15],special:[0,2],specif:[0,4,5,11],specifi:[0,1,2,5,7,10,11,15],speed:14,spent:10,split:[1,15],squar:15,standard:[0,2,8,10,15],start:[2,4,10],stat:10,statist:[1,2,5,10,15],std:[10,15],stderr:[10,15],stem:10,step:[4,15],stop:[2,10,15],stop_aft:10,stopaft:[2,10,15],store:[2,5,10,11,15],str:[5,7,9,10,11],stream:[0,1,2,4,7,9,10,15],stream_nam:[2,7,9,10],string:[0,1,9,15],strongli:0,structur:[0,2],sub:6,subclass:5,submodul:11,subsequ:[2,10],subset:10,successfulli:5,suffix:[10,11,15],suggest:[1,4],suit:[0,13],sum:15,summar:10,superset:0,supplement:10,suppli:2,support:[0,10],suppos:[0,1,15],suppress:0,sure:15,syntax:0,system:2,take:[1,2,7,9,10],taken:[7,10,15],target:0,task:1,tell:15,tensor:1,tensorboard:2,tensorboardhook:0,tensorflow:[5,13,15],termin:[0,2,4,10],test:[0,1,2,4,10,13,15],test_batch:5,test_stream:[1,5,15],than:[2,10],thank:10,thei:[0,1,15],them:[0,1,2,10,15],therefor:[0,4,15],thi:[0,1,2,3,4,5,9,10,11,13,15],thing:[4,15],third:15,those:[4,15],threshold:10,through:[1,4,6,7,10,15],time:[1,10],togeth:[1,6,15],token:0,top:0,topic:15,total:15,track:2,tracker:14,train:[0,1,2,5,7,8,10,11,15],train_batch:5,train_by_stream:7,train_stream:[1,4,5,7,9,10,15],trainabl:11,trainig:2,trainingtermin:[2,10],translat:2,tri:11,trigger:[2,4,10],truth:0,tune:0,tutori:[2,5,14],two:[0,1,2,4,5],txt:13,type:[0,1,7,10,11],typeerror:10,typic:1,unanot:0,unbias:15,under:[1,2,14],understand:15,union:[7,10,11],unit:[1,15],unix:15,unknown:10,unknown_type_act:10,unsupport:10,unus:7,unused_source_act:7,updat:[1,4,5,11],upon:2,usag:[6,15],use:[0,1,2,4,11,13,15],used:[0,1,4,5,8,10,11,15],useful:[1,2,15],useless:1,user:[0,10,13],using:[0,6,13,15],usual:[2,4,5,10,11,15],util:[10,13],valid:[0,1,2,4,10,15],valid_stream:1,valu:[0,2,4,10,15],valueerror:[7,10],variabl:[0,2,5,10,15],variable_nam:10,variable_scop:15,variou:[0,1,8,15],vector:15,verb:10,verbos:6,veri:[0,4,15],verif:15,via:[1,10,14],visual:1,wai:[0,1,6,13],want:15,warn:[0,7,10],wast:1,watch:[2,15],weak:[7,9,10,11],webpag:0,well:[0,1,15],were:10,what:[0,2,4,5,15],when:[0,1,4,5,7,10,11,15],whenc:0,where:[2,4,5,15],wherein:10,whether:[1,4,5,10],which:[0,1,2,4,5,6,9,10,11,13,14,15],whole:[0,1,2,4,5,6,15],whose:1,width:[0,1],wise:1,wish:5,within:10,without:[1,5,10],work:2,workflow:5,world:15,would:[0,1,2,5,8],wrap:15,wrapper:15,write:[2,10],write_csv:10,writecsv:[2,10],writetensorboard:2,written:[2,9,10],wrong:10,xxx:10,y_hat:15,yaml:[0,1,8,9,15],yet:10,yield:[1,15],you:[0,2,5,14,15],your:[2,4,14,15],zero:15,zeroth:7},titles:["Configuration","Dataset","Hooks","Advanced","Main Loop","Model","CLI Reference","<code class=\"docutils literal\"><span class=\"pre\">cxflow</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow.constants</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow.datasets</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow.hooks</span></code>","<code class=\"docutils literal\"><span class=\"pre\">cxflow.models</span></code>","API Reference","Getting Started","cxflow","Tutorial"],titleterms:{"class":[7,9,10,11],The:1,Using:15,addit:1,advanc:3,after_batch:2,after_epoch:2,api:12,architectur:3,basedataset:1,cli:6,conclus:0,configur:[0,2,15],constant:8,contribut:14,cxflow:[6,7,8,9,10,11,14,15],data:1,dataset:[0,1,4,6,9,15],develop:13,event:2,except:10,get:13,hook:[0,2,4,10,15],infer:0,instal:13,integr:4,introduct:15,lazi:1,licens:14,lifecycl:4,loop:[0,4,15],main:[0,4,15],method:1,model:[0,4,5,11,15],philosophi:1,predict:[4,6],process:1,refer:[6,12],regular:2,requir:13,restor:5,resum:6,run:5,start:13,submodul:7,support:14,task:15,train:[4,6],tutori:15,variabl:8}})