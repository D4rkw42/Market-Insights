import os

REMOTE_REPOSITORY = "origin"
REMOTE_BRANCH = "main"

reset_commands = [
    "git reset --hard HEAD"
]

update_commands = [
    "git fetch",
    f"git merge {REMOTE_REPOSITORY}/{REMOTE_BRANCH}"
]

# Reset do projeto para o commit atual (perda de todas as alterações locais)
def reset() -> bool:
    reply = input("[scripts.project] Deseja apagar todas as alterações locais? (Y/N) ")

    if not reply.lower() == "y":
        return False

    for command in reset_commands:
        print(f"[scripts.project] {command}")
        os.system(command)

    return True

# Atualiza o projeto com todas as alterações do repositório de origem
def update():
    # É necessário resetar as modificações locais para fazer o fetch do repositório remoto
    success = reset()

    # Caso as alterações não tenham sito feitas
    if not success:
        return print("[scripts.project] Limpe as modificações locais para atualizar o repositório local.")

    for command in update_commands:
        print(f"[scripts.project] {command}")
        os.system(command)

if __name__ == "__main__":
    update()
