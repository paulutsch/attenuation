"""

    Network Setups - automatically save a parameter setup

    * print_setup
    * save_setup
    * default setups

"""

import os
import numpy as np
from ANNarchy.core.Synapse import Synapse
from ANNarchy.core.Neuron import Neuron
from shutil import copyfile
from importlib import import_module


def print_setup(setup=None, setup_dict=None, start=0, stop=None, N=None):
    """
        formatted print of a setup object

            * setup: path to .py file containing setup object(s)  OR  a setup object itself
            * setup_dict: setup object  OR  name of a setup object contained in setup.py

    """


    if setup is not None:
        # location of setup.py given
        if isinstance(setup, str):

            copyfile(setup, os.path.join(os.getcwd(), setup[-(len(setup.split("/")[-1])):]))

            if setup_dict is not None:
                setup_name = [setup_dict, ]
            else:
                setup_name = [setup[-(len(setup.split('/')[-1])):-3], ]

            try:
                setup_module = __import__(setup[-(len(setup.split("/")[-1])):-3], fromlist=setup_name)
                setup_dict = eval('setup_module.'+setup_name[0])
            except:
                setup_module = __import__(setup[-(len(setup.split("/")[-1])):-3], fromlist=['setup_new',])
                setup_dict = eval('setup_module.setup_new')

            os.remove(os.path.join(os.getcwd(), setup[-(len(setup.split("/")[-1])):]))

        else:
            setup_dict = setup

    if isinstance(setup_dict, dict):

        maxlen = np.max(np.vectorize(len)(list(setup_dict.keys())))
        print("\n")
        i = 0
        flag = False
        if stop is None:
            if N is None:
                stop = len(setup_dict)
            else:
                stop = start + N
        for key in setup_dict:
            if i>=start and i<stop:
                i += 1
                if flag:
                    if isinstance(setup_dict[key], (tuple, list)):
                        paren = ["[", "]"] if isinstance(setup_dict[key], list) else ("(", ")")
                        if isinstance(setup_dict[key][0], (Synapse, Neuron)):
                            contents = paren[0]
                            for x in setup_dict[key]:
                                contents += x.name+', '
                            contents += paren[1]
                            item = key + " "*(maxlen-len(key))+" = "+contents
                        else:
                            contents = paren[0]
                            for x in setup_dict[key]:
                                contents += str(x)+', '
                            contents = contents[:-2]
                            contents += paren[1]
                            item = key + " "*(maxlen-len(key))+" = "+contents
                    else:
                        if not isinstance(setup_dict[key], (Synapse, Neuron)):
                            item = key + " "*(maxlen-len(key))+" = "+str(setup_dict[key])
                        else:
                            item = key + " "*(maxlen-len(key))+" = "+setup_dict[key].name
                    input(item)
                    flag = False
                else:
                    if i % 25 == 0:
                        end_char = ''
                        flag = True
                    else:
                        end_char = '\n'
                    if isinstance(setup_dict[key], (tuple, list)):
                        if isinstance(setup_dict[key][0], (Synapse, Neuron)):
                            paren = ["[", "]"] if isinstance(setup_dict[key], list) else ("(", ")")
                            contents = paren[0]
                            for x in setup_dict[key]:
                                contents += x.name+', '
                            contents += paren[1]
                            print(key + " "*(maxlen-len(key)), "=", contents, end=end_char)
                        else:
                            print(key + " "*(maxlen-len(key)), "=", setup_dict[key], end=end_char)
                    else:
                        if not isinstance(setup_dict[key], (Synapse, Neuron)):
                            print(key + " "*(maxlen-len(key)), "=", setup_dict[key], end=end_char)
                        else:
                            print(key + " "*(maxlen-len(key)), "=", setup_dict[key].name, end=end_char)

    else:

        print("[Error] Setup is not a dict!")

    return setup_dict


def copy_locals(locals_, without={}):
    """
        copies the locals() dictionary

        * without: dictionary of entries that should not be copied

    """
    copy = {}
    for key in locals_.keys():
        if key not in without and locals_[key] is not without and key != 'Out' and key[0] != '_':
            copy[key] = locals_[key]

    return copy


def save_setup(setup, sdir, sname='setup_new'):
    """
        Saves setup to .py file

    """
    f = open(sdir+'/'+sname+'.py', 'w+')
    try:
        f.write("# "+setup['id_message']+"\n\n")
    except:
        pass
    f.write("from ANNarchy import Neuron\n")
    f.write("from ANNarchy import Synapse\n")
    f.write("\nfrom parameters import defParams\n")
    f.write("\n"+sname+" = {}\n\n")

    # only write model definitions
    model_list = []
    for key in setup.keys():
        if isinstance(setup[key], (tuple, list)):
            for entry in setup[key]:
                if isinstance(entry, (Synapse, Neuron)):
                    # write model definitions only once
                    if str(entry.name) not in model_list:
                        model_list.append(str(entry.name))

                        f.write(str(entry.name)+" = "+("Neuron" if isinstance(entry, Neuron) else "Synapse")+"(\n")
                        f.write("    name='"+str(entry.name)+"',\n")
                        f.write('    parameters="""'+str(entry.parameters)+'""",\n')
                        f.write('    equations="""'+str(entry.equations)+'""",\n')
                        f.write("    extra_values=defParams")
                        f.write(")\n\n")
        else:
            if isinstance(setup[key], (Synapse, Neuron)):
                # write model definitions only once
                if str(setup[key].name) not in model_list:
                    model_list.append(str(setup[key].name))

                    f.write(str(setup[key].name)+" = "+("Neuron" if isinstance(setup[key], Neuron) else "Synapse")+"(\n")
                    f.write("    name='"+str(setup[key].name)+"',\n")
                    f.write('    parameters="""'+str(setup[key].parameters)+'""",\n')
                    f.write('    equations="""'+str(setup[key].equations)+'""",\n')
                    f.write("    extra_values=defParams")
                    f.write(")\n\n")

    # only write simple variable declarations
    for key in setup.keys():
        if not isinstance(setup[key], (Synapse, Neuron)):
            # check for nested model definitions
            if isinstance(setup[key], (tuple, list)):
                if isinstance(setup[key][0], (Synapse, Neuron)):
                    paren = ["[", "]"] if isinstance(setup[key], list) else ("(", ")")
                    f.write(sname+"['"+key+"'] = "+paren[0])
                    for x in setup[key]:
                        f.write(x.name+', ')
                    f.write(paren[1]+"\n")
                else:
                    f.write(sname+"['"+key+"'] = "+str(setup[key])+"\n")
            # check for strings to add quotation marks
            elif isinstance(setup[key], str):
                f.write(sname+"['"+key+"'] = '"+str(setup[key])+"'\n")
            else:
                f.write(sname+"['"+key+"'] = "+str(setup[key])+"\n")
        else:
            f.write(sname+"['"+key+"'] = "+str(setup[key].name)+"\n")

    f.close()

# rbf network setups
try:
    from setups.setup_neuron_activity import setup_neuron_activity
except:
    setup_neuron_activity = {}

try:
    from setups.setup_latest import setup_latest
except:
    setup_latest = {}

setup_rbf_default = {}

_rbf_setups = {'default': setup_rbf_default,
               'latest': setup_latest,
               'neuron_test': setup_neuron_activity}
