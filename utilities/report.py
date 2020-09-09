import os
import matplotlib.pyplot as plt
root_folder = os.path.realpath('.')

def print_stats_trend(history, model_type, dataset, embedding_type):
    print(history.history.keys())
    # summarize history for accuracy
    plt.grid(True)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(root_folder+'/report/'+model_type+"_"+embedding_type
        +"_ds"+str(dataset)+"_accuracy_trend.png") 
    plt.close()
    # summarize history for loss
    plt.grid(True)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(root_folder+'/report/'+model_type+"_"+embedding_type
        +"_ds"+str(dataset)+"_loss_trend.png") 
    plt.close()
