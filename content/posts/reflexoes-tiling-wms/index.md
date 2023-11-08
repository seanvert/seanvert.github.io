---
title: "Reflexões sobre tiling window managers"
date: 2023-11-07T23:02:45-03:00
draft: true
description: "Pensamentos sobre como tiling window managers me afetaram"
tags: ["reflexões", "tiling", "mental health"]
series: ["Tiling WM"]
serie_order: 1
---
Este post será o primeiro de uma série que eu pretendia escrever já faz algum tempo sobre tiling window managers. Não vou comentar sobre o que eles são, porque já que existem outros recursos bons falando disso.

[Window Manager - Arch Linux Wiki (Português)](https://wiki.archlinux.org/title/Window_manager_(Portugu%C3%AAs))

[Tiling Window Managers - Wikipedia (Inglês)](https://en.wikipedia.org/wiki/Tiling_window_manager)

## Pontos Fortes
- interface personalizável
Você consegue com alguma facilidade montar o seu próprio ambiente dee trabalho, e focar em fazer o que importa ao invés de perder tempo usando alt-tab.
- unir aplicações diferentes
Se você não tem uma 'IDE' (integrated development environment), um tiling WM é um bom ponto de partida para montar algo nessas linhas, ou que pelo menos no sentido da interface fique com uma funcionalidade parecida.
- múltiplos monitores
É possível criar configurações pré-definidas para cada monitor e economizar tempo com deixar cada janela no lugar dela.
- fácil de trocar posição de janelas
Isso é bem útil se você precisa constantemente trocar o foco entre janelas.
- você não precisa escolher entre um ou outro se não quiser
Pode usar os tiling wm junto com xfce, gnome, kde e etc. E ter ´o melhor de ambos mundos´.

## O que pode ser ruim

- aparentemente é muito ruim para o ´attention span´ (se você souber de alguma palavra que fica legal aqui e quiser me mandar em alguma reede social, fique a vontade). Supondo que você use esse tipo de interface para tudo, você vai rapidamente perceber que vai precisar ter muito mais disciplina, por o que te distraia na internet antes, agora está multiplicado por 2-3.
- curva de aprendizado acentuada
não é o tipo de coisa que eu aconselho você começar a usar quando estiver apertado para entregar algo no prazo. Acho que nas primeiras semanas é normal ter alguma dificuldade para se acostumar.
- não é tão vantajoso assim com monitores pequenos
Talvez você tire mais de um notebook com um monitor pequeno, usando múltiplas áreas de trabalho.
- alguns exigem algumas noções de programação para configurar
o que também pode ser uma vantagem dependendo da pessoa. Pode ser uma boa oportunidade para sair do 'hello world' e começar a programar algo realmente útil. 

## Funcionalidades comuns

- Scrachpads
- Layouts Dinâmicos

## Dicas para começar a usar algum

- use um papel de parede com os atalhos de teclado
- ou alternativamente, faça um scratchpad com o feh com uma imagem com os seus atalhos
{{< img src="/img/i3.webp" caption="Exemplo do que você não deve fazer se quiser se focar" >}}
- se perguntar se faz sentido você usar isso
por mais que as screenshots no r/unixporn sejam legais, acho que vale a pena se perguntar se você vai ter alguma utilidade pra esse tipo de coisa porque ele *vai* te atrapalhar até você se acostumar. E não sei até que ponto é o tipo de coisa legal pra quem teme TDAH, mas também vou comentar algo sobre foco na próxima.
- você pode copiar uma janela para todas as áreas de trabalho
isso é interessante pra você não acabar procrastinando algo que deveria estar fazendo enquanto estiver pesquisando alguma coisa na internet e não correr o risco de se ´perder´ porque o que você precisa fazer estará *sempre* visível.
- considerando que você vai passar boa parte do tempo só usando o computador
acho justo presumir que a maior parte do tempo o seu layout será o 'maximizado' mesmo. E considerando a minha experiência com o XMonad, acho que não compensa usar mais do que 4 layouts diferentes. Talvez o ideal sejam dois, e você alterne entre isso e o maximizado por área de trabalho, ainda não testei isso.
- dependendo do que você for usar a documentação pode ser um pouco intimidadora ou ruim esse é o caso do window manager que eu uso (XMonad) a documentação não é das melhores. Nesse caso, eu recomendo olhar o código fonte que costuma ser bem documentado e fazer mais sentido.
- se você não souber programar, ou tiver alguma familiaridade com alguma linguagem

[Comparação entre diferentes Tiling WMs (Inglês)](https://wiki.archlinux.org/title/Comparison_of_tiling_window_managers)

## Programas externos recomendados
você só vai precisar instalar estes jogos se não for usar ele junto com algum desktop environment
- rofi
    * faça um menu para selecionar as janelas abertas
- tudo do xfce
- barras de status
    * polybar
    * lemonbar
- system tray
    * stalonetray


