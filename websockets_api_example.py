#This is an example that uses the websockets api to know when a workflow execution is done
#Once the workflow execution is done it downloads the images using the /history endpoint

import fnmatch
import os
import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import random
import re

#samplers = ["dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m",  "dpmpp_2m_sde",  "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu"]
samplers = ["euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m",  "dpmpp_2m_sde",  "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ddim", "uni_pc", "uni_pc_bh2"]
schedulers = ["normal"]
#steps = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
steps = 50
next_step = 1
vae_index = ["4", 2]


def create_clip_text_encode(params):
    return {
        params.get('key'): {
            "inputs": {
                "text": params.get('text'),
                "clip_model": params.get('clip_model')
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": params.get('title')
            }
        }
    }
def create_KSampler(params):
    return {
        params.get('key'): {
        "inputs": {
            "add_noise": params.get('add_noise'),
            "noise_seed": params.get('noise_seed'),
            "steps": params.get('steps'),
            "cfg": params.get('cfg'),
            "sampler_name": params.get('sampler_name'),
            "scheduler": params.get('scheduler'),
            "start_at_step": params.get('start_at_step'),
            "end_at_step": params.get('end_at_step'),
            "return_with_leftover_noise": params.get('return_with_leftover_noise'),
            "model": params.get('model'),
            "positive": params.get('positive'),
            "negative": params.get('negative'),
            "latent_image": params.get('latent_image')
        },
        "class_type": "KSamplerAdvanced",
        "_meta": {
            "title": params.get('title')
        }
    }
    }
def create_vae_decoder(params):
    return {
        params.get('key'): {
            "inputs": {
                "samples": params.get('samples'),
                "vae": params.get('vae')
            },
            "class_type": "VAEDecode",
            "_meta": {
                "title": params.get('title')
            }
        }
    }
def create_save_image(params):
    return {
        params.get('key'): {
            "inputs": {
                "filename_prefix": params.get('filename_prefix'),
                "images": params.get('images')
            },
            "class_type": "SaveImage",
            "_meta": {
                "title": params.get('title')
            }
        }
    }




server_address = "127.0.0.1:8188"

client_id = str(uuid.uuid4())

def generate_random_number(num_digits):
    range_start = 10**(num_digits - 1)
    range_end = (10**num_digits) - 1
    return random.randint(range_start, range_end)

def queue_workflow(workflow):
    p = {"prompt": workflow, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def fetch_queue():
    url = "http://{}/queue".format()
    req = urllib.request.Request(url)
    return json.loads(urllib.request.urlopen(req).read())

def interrupt_running_process():
    data = json.dumps({"client_id": client_id}).encode('utf-8')
    req = urllib.request.Request("http://{}/interrupt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())
        
def display_and_cancel():
    try:
        queue = fetch_queue()
        running = queue.get("queue_running", [])
        pending = queue.get("queue_pending", [])

        print("Running items:", running)
        print("Pending items:", pending)

        # Cancel the first running item
        if running:
            response = interrupt_running_process()
            print("Cancelled the first running item:", response)
        """
        # Cancel all pending items
        for process in pending:
            process_id = process.get("id")
            if process_id:
                response = cancel_pending_process(server_address, client_id, process_id)
                print(f"Cancelled pending item with ID {process_id}:", response)
        """
    except Exception as e:
        print("Error fetching queue or cancelling items:", str(e))

def get_images(ws, workflow):
    workflow_id = queue_workflow(workflow)['prompt_id']
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == workflow_id:
                    break #Execution is done
        else:
            continue #previews are binary data

def read_and_parse_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = f.read()
    return json.loads(data)

def set_elements(workflow, sampler, scheduler, steps, current_step, next_step):
    params = {
    'key': current_step+10,
    'title': 'KSampler{}'.format(current_step),
    "noise_seed": 5,
    "add_noise": "enable",
    'steps': steps,
    'cfg': 8,
    'sampler_name': sampler,
    'scheduler': scheduler,
    'start_at_step': 0,
    'end_at_step': current_step+next_step,
    'return_with_leftover_noise': "disable",
    'model': ["4",0],
    "positive": ["6",0],
    "negative": ["7",0],
    #"latent_image": [str(current_step+10-next_step),0]
    "latent_image": ["5", 0]
    }
    if current_step == 0:
        params["latent_image"] = ["5", 0]
    k_sampler = create_KSampler(params)
    workflow.update(k_sampler)

    params = {
    'key': current_step+100,
    'samples': [str(current_step+10), 0],
    'vae': vae_index,
    'title': "VAE Decode"
    }
    vae_decoder = create_vae_decoder(params)
    workflow.update(vae_decoder)

    params = {
    'key': current_step+200,
    'filename_prefix': "{}_{}_{}".format(sampler, scheduler, current_step+next_step),
    'images': [str(current_step+100),0],
    'title': "Save" + str(current_step+next_step)
    }
    save_image = create_save_image(params)
    workflow.update(save_image)

def check_images(directory, prefix):
    # List all files in the directory
    files = os.listdir(directory)
    # Filter the list of files for those that match the prefix
    matching_files = fnmatch.filter(files, prefix+'*')
    # If no matching files are found, return False
    if not matching_files:
        return False
    # Otherwise, return True
    return True

def set_workflow():
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    for sampler in samplers:
        for scheduler in schedulers:
            workflow = read_and_parse_json("workflow_api_delna.json")
            images_exist = check_images('/home/sivar/ComfyUI-master/output', '{}_{}_{}'.format(sampler, scheduler, steps))
            if images_exist:
                continue
            for i in range(0, steps, next_step):
                images_exist = check_images('/home/sivar/ComfyUI-master/output', '{}_{}_{}'.format(sampler, scheduler, i+next_step))
                if images_exist:
                    continue
                set_elements(workflow, sampler, scheduler, steps, i, next_step)
            get_images(ws, workflow)
            #with open('output_file.json', 'w') as file:
                #json.dump(workflow, file, indent=2)



    

#set the text workflow for our positive CLIPTextEncode
#workflow["6"]["inputs"]["text"] = "masterpiece best quality man"

#set the seed for our KSampler node
#workflow["3"]["inputs"]["seed"] = 5


#get_images(ws, workflow)

set_workflow()
