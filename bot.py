import discord
from discord.ext import commands
from typing import Tuple
import os,re,argparse
import sys
import torch
import time
import json
from pathlib import Path
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama import ModelArgs, Transformer, Tokenizer, LLaMA

def log(str):
    a=open("log.txt","ab")
    a.write(str.encode())
    a.write('\n'.encode())
    a.close()

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

def init(ckpt_dir,tokenizer_path):
    max_seq_len = 512
    max_batch_size = 32
    global generator
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

intents = discord.Intents.default()
intents.members = True
intents.typing = True
intents.presences = True
intents.message_content = True

bot = commands.Bot(command_prefix="!", help_command=None,intents=intents)

@bot.command()
async def info(ctx):
    print('info')
    await ctx.send(ctx.guild)
    await ctx.send(ctx.author)

@bot.event
async def on_ready() -> None:
    msg=f"Bot {bot.user} waking up."
    print(msg)
    log(msg)
    await bot.change_presence(activity=discord.Game(name="global thermonuclear war")) 

sem = 0 # global semaphore

@bot.event
async def on_message(message):
    #Default config values
    temperature= 0.9
    top_p= 0.75
    max_len=256
    repetition_penalty_range=1024
    repetition_penalty=1.15
    repetition_penalty_slope=0.7
    #semaphore
    global sem
    if message.author == bot.user:
        return

    if (sem==1):
        time.sleep(5)
    sem=1
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    botid=("<@%d>" % bot.user.id)
    if message.content.startswith(botid):
        query = message.content[len(botid):].strip()
        origquery=query
        query=query[:1024] #limit query lenght
        jsonEnd=query.find('}')
        rawpos=query.find('raw')
        if (jsonEnd > rawpos):
            jsonEnd=0 # this is not configuration
        try:
            if (jsonEnd>0): # json config present, parse
                    config=query[:jsonEnd+1]
                    query=query[jsonEnd+1:].strip()
                    config=json.loads(config)
                    if not (config.get('temperature') is None):
                        temperature=float(config['temperature'])
                    if not (config.get('top_p') is None):
                        top_p=float(config['top_p'])
                    if not (config.get('max_len') is None):
                        max_len=int(config['max_len'])
                        if (max_len>2048): max_len=2048
                    if not (config.get('repetition_penalty_range') is None):
                        repetition_penalty_range=int(config['repetition_penalty_range'])
                    if not (config.get('repetition_penalty') is None):
                        repetition_penalty_range=float(config['repetition_penalty'])
                    if not (config.get('repetition_penalty_slope') is None):
                        repetition_penalty_range=float(config['repetition_penalty_slope'])
        except Exception as e:
                if local_rank == 0:
                    msg = f"{message.author.mention} Error parsing the Json config: %s" % str(e)
                    log(msg)
                    await message.channel.send(msg)
                sem=0
                return

        if (query.startswith('raw ')): # Raw prompt
                query = query[4:]
        else: # Wrap prompt in question
                query ='The answer for "%s" would be: ' % query
        prompts = [query]
        print(prompts)
        torch.cuda.empty_cache()
        async with message.channel.typing():
            try:
                results = generator.generate(prompts, max_gen_len=max_len, temperature=temperature, top_p=top_p, repetition_penalty_range=repetition_penalty_range,repetition_penalty=repetition_penalty,repetition_penalty_slope=repetition_penalty_slope)
            except: 
                if local_rank == 0:
                    await message.channel.send(msg)
                sem=0
                return
        if local_rank == 0:
            sem=0
            return
        log("---"+str(origquery))

        for result in results:
            msg = f"{message.author.mention} %s" % result
            log(msg)
            if len(msg)>1500:
                for i in range(0,len(msg),1500):
                    await message.channel.send(msg[i:i+1500])
            else:
                await message.channel.send(msg)
    sem=0

#Argument parser
parser = argparse.ArgumentParser(description='BasedGPT AI discord chat, (C) Cybergaucho 2023 @ortegaalfredo')
parser.add_argument('--ckpt_dir', type=str,required=True,help='Model directory')
parser.add_argument('--tokenizer_path', type=str, required=True, help='Tokenizer path')
args=parser.parse_args()

# Init AI
init(ckpt_dir=args.ckpt_dir,tokenizer_path=args.tokenizer_path)
# Connect bot
token=open('discordtoken.txt').read().strip()
bot.run(token)
