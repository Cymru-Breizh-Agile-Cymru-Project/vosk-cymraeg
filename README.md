# Vosk Cymraeg / Welsh Vosk

Little package currently under development

## Fetching datasets

To fetch the datasets from HuggingFace, run the following script:
```bash
uv run fetch
```
Note that you need to have the enviroment variable HF_TOKEN set in order to read from HuggingFace. This can be set using a .env file in the root directory of the project.

## Running the training environment
This repository supports both [Dev Containers](https://containers.dev/) and [Docker compose](https://docs.docker.com/compose/) and both use the same Dockerfile.

To use Docker compose run and attach to the training environment run the following command:
`docker compose run --rm training_env`. The current Docker compose file is set up with a default user, UID, and GID. You should change these to the same as the host machine to ensure that files made within the environment has the correct permissions and owners. You can find the current users UID and GID by using the `id` command on Linux.

## Datasets Used

### Transcribed Speech
|Name|Author|Source|
|---|---|---|
| Banc Trawsgrifiadau Bangor | Language Technologies Unit, Bangor University | [link](https://huggingface.co/datasets/techiaith/banc-trawsgrifiadau-bangor) |
| Lleisiau ARFOR | Cymen in collaboration with the Language Technologies Unit, Bangor University | [link](https://huggingface.co/datasets/cymen-arfor/lleisiau-arfor)
| Common Voice | Mozilla Foundation | [link](https://commonvoice.mozilla.org) |

### Pronunciation Dictionaries
|Name|Author|Source|
|---|---|---|
| Geiriadur Ynganu Bangor | Language Technologies Unit, Bangor University | [link](https://github.com/techiaith/geiriadur-ynganu-bangor) |

In addition the resources listed above, the final pronunciation dictionary contains entires from [Lecsicon Cymraeg Bangor](https://github.com/techiaith/lecsicon-cymraeg-bangor) that has been converted using an [updated version](https://github.com/Cymru-Breizh-Agile-Cymru-Project/vosk-cymraeg/blob/main/src/vosk_cymraeg/llef_py3.py) of [welsh-lts](https://github.com/techiaith/welsh-lts/blob/master/llef.py).

## Maintainers

This repository is being created and maintained by:
- [Gweltaz Duval-Guennoc](https://github.com/gweltou)
- [Dewi Bryn Jones](https://github.com/DewiBrynJones)
- [Preben Vangberg](https://github.com/prvInSpace)
- [Sasha Wanasky](https://github.com/wanasash)

This work is being funded by the Welsh Government through their [Agile Cymru](https://www.gov.wales/agile-cymru-guidance) programme.