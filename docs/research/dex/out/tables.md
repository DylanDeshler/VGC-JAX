### T1. Slot-mass coverage curves (entities needed for 90/95/99/99.9% of slot mass; total observed)

| format | battles | species (incl. megas) | moves | abilities | items (non-stone) | mega stones | mega slot share |
|---|---|---|---|---|---|---|---|
| bo1_0 | 1631943 | 92/145/236/304 (333) | 121/174/290/414 (510) | 54/77/128/178 (214) | 20/30/54/72 (85) | 32/48/68/79 (81) | 0.277 |
| bo1_1500 | 1631943 | 73/114/208/284 (316) | 107/150/255/383 (510) | 47/66/112/164 (212) | 17/25/48/68 (85) | 26/39/63/77 (81) | 0.282 |
| bo1_1630 | 1631943 | 60/90/178/258 (294) | 98/135/230/349 (508) | 41/57/97/149 (206) | 16/22/43/64 (84) | 23/31/56/75 (80) | 0.284 |
| bo1_1760 | 1631943 | 58/84/167/247 (277) | 95/131/223/340 (502) | 40/55/94/144 (203) | 16/22/42/63 (84) | 22/29/53/74 (80) | 0.285 |
| bo3_0 | 223477 | 54/82/175/266 (302) | 93/128/224/371 (504) | 39/54/97/156 (209) | 15/20/43/66 (84) | 21/31/57/75 (81) | 0.294 |
| bo3_1500 | 223477 | 47/65/135/223 (259) | 84/114/190/318 (491) | 34/47/80/133 (195) | 14/19/36/59 (84) | 19/25/47/70 (80) | 0.296 |
| bo3_1630 | 223477 | 42/56/104/176 (212) | 79/104/166/262 (478) | 32/42/69/110 (174) | 14/18/30/49 (82) | 17/22/38/58 (69) | 0.295 |
| bo3_1760 | 223477 | 39/52/92/158 (191) | 77/100/157/242 (473) | 31/40/65/102 (167) | 14/18/28/48 (81) | 17/21/33/52 (65) | 0.295 |

### T2. Candidate dexes: size, handler cost, team coverage

| dex | species | moves | abilities | items | CUSTOM ents (incl. conds) | conditions | distinct handler templates | handler fns | team cov 1760 Bo3 | 1760 Bo1 | all Bo3 | all Bo1 | real Bo3 replay teams (strict) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mass95_bo3_1760 | 50+20M | 100 | 40 | 36 (17 stones) | 106 | 42 | 212 | 285 | 0.349 | 0.130 | 0.184 | 0.052 | 0.003 |
| mass99_bo3_1760 | 83+34M | 157 | 65 | 60 (31 stones) | 154 | 51 | 292 | 402 | 0.707 | 0.395 | 0.464 | 0.213 | 0.125 |
| usage_bo3_1760>=0.001 | 97+39M | 231 | 107 | 84 (39 stones) | 230 | 59 | 403 | 572 | 0.893 | 0.655 | 0.700 | 0.457 | 0.763 |
| usage_bo3_1760>=0.0001 | 165+61M | 314 | 138 | 120 (61 stones) | 300 | 67 | 509 | 733 | 0.992 | 0.845 | 0.839 | 0.691 | 0.900 |
| usage_bo3_1760>=0.0001+bo1_0>=0.0005 | 236+81M | 426 | 177 | 157 (80 stones) | 396 | 85 | 633 | 940 | 0.999 | 0.994 | 0.978 | 0.949 | 0.980 |
| cost_aware_tau=0.01 | 293+82M | 377 | 110 | 142 (81 stones) | 220 | 57 | 369 | 622 | 0.899 | 0.778 | 0.787 | 0.620 | 0.820 |
| cost_aware_tau=0.003 | 293+82M | 423 | 145 | 151 (81 stones) | 311 | 75 | 519 | 806 | 0.978 | 0.946 | 0.924 | 0.866 | 0.957 |
| cost_aware_tau=0.001 | 293+82M | 454 | 172 | 160 (81 stones) | 376 | 83 | 603 | 919 | 0.997 | 0.991 | 0.975 | 0.957 | 0.994 |
| cost_aware_tau=0.0005 | 293+82M | 469 | 183 | 161 (81 stones) | 402 | 86 | 632 | 963 | 0.999 | 0.995 | 0.984 | 0.972 | 0.997 |
| cost_aware_tau=0.0003 | 293+82M | 481 | 192 | 161 (81 stones) | 423 | 90 | 659 | 996 | 1.000 | 0.998 | 0.992 | 0.985 | 0.999 |
| cost_aware_tau=0.0002 | 293+82M | 493 | 200 | 162 (81 stones) | 445 | 93 | 690 | 1035 | 1.000 | 0.999 | 0.996 | 0.992 | 0.999 |
| cost_aware_tau=0.0001 | 293+82M | 502 | 206 | 163 (81 stones) | 462 | 100 | 712 | 1069 | 1.000 | 1.000 | 0.998 | 0.997 | 0.999 |
| cost_aware_tau=5e-05 | 293+82M | 505 | 208 | 165 (81 stones) | 469 | 100 | 726 | 1085 | 1.000 | 1.000 | 0.999 | 0.999 | 1.000 |
| cost_aware_tau=2e-05 | 293+82M | 507 | 214 | 165 (81 stones) | 477 | 100 | 735 | 1095 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| cost_aware_tau=0 | 293+82M | 510 | 215 | 166 (81 stones) | 483 | 102 | 747 | 1107 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| cost_aware_bo3_1760_only_tau=0.003 | 293+82M | 361 | 114 | 139 (81 stones) | 208 | 54 | 351 | 592 | 0.886 | 0.691 | 0.721 | 0.482 | 0.804 |
| cost_aware_bo3_1760_only_tau=0.001 | 293+82M | 384 | 129 | 143 (81 stones) | 248 | 60 | 413 | 668 | 0.956 | 0.828 | 0.834 | 0.661 | 0.902 |
| cost_aware_bo3_1760_only_tau=0.0003 | 293+82M | 400 | 142 | 146 (81 stones) | 282 | 65 | 468 | 736 | 0.983 | 0.882 | 0.875 | 0.751 | 0.930 |
| cost_aware_bo3_1760_only_tau=0.0001 | 293+82M | 424 | 150 | 149 (81 stones) | 312 | 69 | 514 | 800 | 0.996 | 0.925 | 0.916 | 0.834 | 0.955 |
| full_legal_pool (v2) | 293+82M | 510 | 215 | 166 (81 stones) | 483 | 102 | 747 | 1107 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### T3-v1. Classification counts (v1)

| kind | n | DATA_ONLY | DATA_PLUS_COMMON | CUSTOM |
|---|---|---|---|---|
| move | 454 | 260 | 65 | 129 |
| ability | 172 | 6 | 23 | 143 |
| item | 160 | 7 | 124 | 29 |
| condition | 83 | 1 | 7 | 75 |

### T3-legal. Classification counts (legal)

| kind | n | DATA_ONLY | DATA_PLUS_COMMON | CUSTOM |
|---|---|---|---|---|
| move | 510 | 260 | 76 | 174 |
| ability | 215 | 6 | 24 | 185 |
| item | 166 | 7 | 124 | 35 |
| condition | 102 | 2 | 11 | 89 |

### T4-v1. Hook frequency (top 45 of 118 hook names; templates = distinct branches after parametric dedup)

| hook | entities | distinct templates | moves | abilities | items | conditions |
|---|---|---|---|---|---|---|
| onTakeItem | 82 | 3 | 0 | 1 | 81 | 0 |
| onStart | 81 | 56 | 0 | 19 | 8 | 54 |
| onHit | 62 | 50 | 56 | 2 | 0 | 4 |
| onBasePower | 52 | 26 | 10 | 15 | 20 | 7 |
| onTryHit | 43 | 36 | 21 | 15 | 0 | 7 |
| basePowerCallback | 25 | 20 | 25 | 0 | 0 | 0 |
| onEnd | 25 | 17 | 0 | 6 | 0 | 19 |
| onModifyMove | 24 | 22 | 9 | 9 | 2 | 4 |
| onSourceModifyDamage | 24 | 7 | 0 | 5 | 17 | 2 |
| onDamagingHit | 24 | 20 | 0 | 19 | 2 | 3 |
| onEat | 24 | 5 | 0 | 0 | 24 | 0 |
| onTry | 23 | 18 | 23 | 0 | 0 | 0 |
| onResidual | 23 | 20 | 0 | 6 | 2 | 15 |
| onUpdate | 20 | 12 | 0 | 10 | 8 | 2 |
| onBeforeMove | 16 | 16 | 0 | 0 | 0 | 16 |
| durationCallback | 16 | 6 | 0 | 0 | 0 | 16 |
| onPrepareHit | 13 | 8 | 10 | 3 | 0 | 0 |
| onModifyAtk | 11 | 7 | 0 | 9 | 1 | 1 |
| onRestart | 11 | 11 | 0 | 0 | 0 | 11 |
| onTryBoost | 10 | 5 | 0 | 10 | 0 | 0 |
| onSwitchIn | 10 | 10 | 0 | 3 | 0 | 7 |
| onSideStart | 10 | 4 | 0 | 0 | 0 | 10 |
| onFieldStart | 10 | 5 | 0 | 0 | 0 | 10 |
| onFieldEnd | 10 | 1 | 0 | 0 | 0 | 10 |
| onModifyType | 9 | 6 | 4 | 5 | 0 | 0 |
| onTryMove | 9 | 6 | 8 | 0 | 0 | 1 |
| onDisableMove | 9 | 6 | 2 | 0 | 0 | 7 |
| onModifySpA | 9 | 6 | 0 | 7 | 1 | 1 |
| onModifySpe | 9 | 8 | 0 | 4 | 2 | 3 |
| onImmunity | 9 | 3 | 0 | 8 | 0 | 1 |
| onTryAddVolatile | 9 | 5 | 0 | 6 | 0 | 3 |
| onSetStatus | 9 | 5 | 0 | 7 | 0 | 2 |
| onAfterSubDamage | 8 | 6 | 7 | 0 | 1 | 0 |
| onDamage | 8 | 8 | 0 | 6 | 1 | 1 |
| onAfterHit | 6 | 4 | 6 | 0 | 0 | 0 |
| onTryImmunity | 6 | 5 | 6 | 0 | 0 | 0 |
| onModifyCritRatio | 6 | 4 | 0 | 2 | 2 | 2 |
| damageCallback | 5 | 4 | 5 | 0 | 0 | 0 |
| onAfterMoveSecondary | 5 | 5 | 0 | 2 | 2 | 1 |
| onWeather | 5 | 5 | 0 | 4 | 0 | 1 |
| onMoveFail | 4 | 2 | 4 | 0 | 0 | 0 |
| onSourceModifyAccuracy | 4 | 4 | 0 | 2 | 2 | 0 |
| onAnyModifyDamage | 4 | 3 | 0 | 1 | 0 | 3 |
| onModifyDef | 4 | 4 | 0 | 3 | 0 | 1 |
| onSourceModifyAtk | 4 | 3 | 0 | 4 | 0 | 0 |

v1: 118 hook names collapse to 93 events after stripping Foe/Ally/Source/Any prefixes.


### T4-legal. Hook frequency (top 45 of 135 hook names; templates = distinct branches after parametric dedup)

| hook | entities | distinct templates | moves | abilities | items | conditions |
|---|---|---|---|---|---|---|
| onStart | 97 | 65 | 0 | 26 | 8 | 63 |
| onHit | 83 | 70 | 76 | 2 | 0 | 5 |
| onTakeItem | 83 | 4 | 0 | 2 | 81 | 0 |
| onBasePower | 55 | 29 | 11 | 17 | 20 | 7 |
| onTryHit | 49 | 42 | 26 | 16 | 0 | 7 |
| onEnd | 31 | 19 | 0 | 7 | 0 | 24 |
| onTry | 30 | 24 | 30 | 0 | 0 | 0 |
| onSourceModifyDamage | 30 | 11 | 0 | 7 | 18 | 5 |
| onResidual | 30 | 26 | 0 | 10 | 2 | 18 |
| basePowerCallback | 29 | 23 | 29 | 0 | 0 | 0 |
| onModifyMove | 28 | 26 | 10 | 11 | 2 | 5 |
| onDamagingHit | 28 | 24 | 0 | 23 | 2 | 3 |
| onEat | 28 | 8 | 0 | 0 | 28 | 0 |
| onUpdate | 25 | 16 | 0 | 11 | 11 | 3 |
| durationCallback | 19 | 6 | 0 | 0 | 0 | 19 |
| onBeforeMove | 16 | 16 | 0 | 0 | 0 | 16 |
| onTryMove | 14 | 8 | 13 | 0 | 0 | 1 |
| onPrepareHit | 13 | 8 | 10 | 3 | 0 | 0 |
| onModifyAtk | 13 | 9 | 0 | 11 | 1 | 1 |
| onFieldStart | 13 | 8 | 0 | 0 | 0 | 13 |
| onTryBoost | 12 | 7 | 0 | 12 | 0 | 0 |
| onModifySpA | 12 | 8 | 0 | 10 | 1 | 1 |
| onImmunity | 12 | 5 | 0 | 8 | 0 | 4 |
| onRestart | 12 | 12 | 0 | 0 | 0 | 12 |
| onFieldEnd | 12 | 2 | 0 | 0 | 0 | 12 |
| onDamage | 11 | 11 | 0 | 8 | 2 | 1 |
| onModifySpe | 11 | 10 | 0 | 6 | 2 | 3 |
| onSetStatus | 11 | 7 | 0 | 8 | 0 | 3 |
| onSideStart | 11 | 4 | 0 | 0 | 0 | 11 |
| onModifyType | 10 | 7 | 4 | 5 | 0 | 1 |
| onDisableMove | 10 | 7 | 2 | 0 | 0 | 8 |
| onSwitchIn | 10 | 10 | 0 | 3 | 0 | 7 |
| onTryAddVolatile | 10 | 6 | 0 | 6 | 0 | 4 |
| onAfterHit | 8 | 6 | 8 | 0 | 0 | 0 |
| onAfterSubDamage | 8 | 6 | 7 | 0 | 1 | 0 |
| onTryImmunity | 7 | 6 | 7 | 0 | 0 | 0 |
| damageCallback | 7 | 5 | 7 | 0 | 0 | 0 |
| onTrapPokemon | 7 | 4 | 0 | 1 | 1 | 5 |
| onModifyCritRatio | 6 | 4 | 0 | 2 | 2 | 2 |
| onAfterMove | 5 | 4 | 3 | 0 | 0 | 2 |
| onAfterMoveSecondary | 5 | 5 | 0 | 2 | 2 | 1 |
| onWeather | 5 | 5 | 0 | 4 | 0 | 1 |
| onTerrainChange | 5 | 2 | 0 | 1 | 4 | 0 |
| onModifyAccuracy | 5 | 5 | 0 | 3 | 1 | 1 |
| onSideEnd | 5 | 1 | 0 | 0 | 0 | 5 |

legal: 135 hook names collapse to 100 events after stripping Foe/Ally/Source/Any prefixes.


### T5. v1 hook -> entity lists, by entity type (paths; `condition.*` = handlers of the entity's own condition)


#### move (24 hook paths)

| hook | n | templates | entities |
|---|---|---|---|
| onHit | 47 | 38 | acupressure, afteryou, allyswitch, banefulbunker, batonpass, bellydrum, bugbite, clangoroussoul, clearsmog, copycat, curse, defog, detect, endure, entrainment, healpulse, instruct, kingsshield, magicpowder, moonlight, morningsun, painsplit, partingshot, pluck, pollenpuff, protect, psychup, quash, rest, shedtail, shellsidearm, simplebeam, skillswap, sleeptalk, soak, speedswap, spikyshield, steelroller, strengthsap, substitute, ... (+7) |
| basePowerCallback | 25 | 20 | acrobatics, assurance, avalanche, beatup, eruption, grassknot, gyroball, hardpress, heatcrash, heavyslam, hex, infernalparade, lastrespects, lowkick, payback, powertrip, ragefist, risingvoltage, round, stompingtantrum, storedpower, temperflare, tripleaxel, watershuriken, waterspout |
| onTry | 23 | 18 | aurawheel, auroraveil, clangoroussoul, counter, fakeout, firstimpression, followme, futuresight, lastresort, mirrorcoat, noretreat, poltergeist, quickguard, ragepowder, rest, round, sleeptalk, snore, steelroller, stockpile, suckerpunch, upperhand, wideguard |
| onTryHit | 21 | 17 | brickbreak, clangoroussoul, curse, disable, entrainment, grassknot, healingwish, heatcrash, heavyslam, helpinghand, lowkick, pollenpuff, poltergeist, psychicfangs, ragingbull, revivalblessing, shedtail, simplebeam, substitute, worryseed, yawn |
| onPrepareHit | 10 | 5 | allyswitch, banefulbunker, destinybond, detect, endure, fling, kingsshield, protect, shellsidearm, spikyshield |
| onBasePower | 10 | 7 | barbbarrage, expandingforce, facade, ficklebeam, knockoff, lashout, mistyexplosion, solarbeam, solarblade, venoshock |
| onModifyMove | 9 | 8 | beatup, blizzard, curse, expandingforce, hurricane, shellsidearm, terrainpulse, thunder, weatherball |
| onTryMove | 8 | 5 | burnup, doubleshock, electroshot, phantomforce, pollenpuff, skyattack, solarbeam, solarblade |
| onAfterSubDamage | 7 | 5 | ceaselessedge, icespinner, mortalspin, rapidspin, shellsidearm, steelroller, stoneaxe |
| onAfterHit | 6 | 4 | ceaselessedge, icespinner, knockoff, mortalspin, rapidspin, stoneaxe |
| onTryImmunity | 6 | 5 | endeavor, leechseed, octolock, switcheroo, trick, worryseed |
| secondary.onHit | 5 | 4 | alluringvoice, burningjealousy, direclaw, throatchop, triattack |
| damageCallback | 5 | 4 | counter, endeavor, finalgambit, mirrorcoat, superfang |
| onModifyType | 4 | 4 | aurawheel, ragingbull, terrainpulse, weatherball |
| onMoveFail | 4 | 2 | axekick, highjumpkick, steelbeam, supercellslam |
| self.onHit | 4 | 2 | batonpass, burnup, doubleshock, shedtail |
| priorityChargeCallback | 2 | 2 | chillyreception, focuspunch |
| beforeTurnCallback | 2 | 1 | counter, mirrorcoat |
| onDisableMove | 2 | 1 | fakeout, firstimpression |
| onHitField | 2 | 2 | haze, perishsong |
| onHitSide | 2 | 1 | quickguard, wideguard |
| beforeMoveCallback | 1 | 1 | focuspunch |
| onEffectiveness | 1 | 1 | freezedry |
| onModifyPriority | 1 | 1 | grassyglide |

#### ability (72 hook paths)

| hook | n | templates | entities |
|---|---|---|---|
| onStart | 19 | 14 | cloudnine, curiousmedicine, drizzle, drought, electricsurge, fairyaura, frisk, grassysurge, hospitality, intimidate, moldbreaker, pressure, psychicsurge, sandstream, screencleaner, snowwarning, supremeoverlord, trace, unnerve |
| onDamagingHit | 19 | 16 | cursedbody, cutecharm, electromorphosis, flamebody, gooey, illusion, innardsout, justified, mummy, poisonpoint, roughskin, seedsower, spicyspray, stamina, static, thermalexchange, toxicdebris, wanderingspirit, weakarmor |
| onBasePower | 15 | 10 | aerilate, dragonize, ironfist, megalauncher, pixilate, punkrock, reckless, refrigerate, sandforce, sharpness, sheerforce, strongjaw, supremeoverlord, technician, toughclaws |
| onTryHit | 15 | 12 | bulletproof, dryskin, eartheater, flashfire, goodasgold, lightningrod, magicbounce, oblivious, overcoat, sapsipper, soundproof, sturdy, telepathy, voltabsorb, waterabsorb |
| onTryBoost | 10 | 5 | clearbody, hypercutter, illuminate, innerfocus, keeneye, mirrorarmor, oblivious, owntempo, scrappy, whitesmoke |
| onUpdate | 10 | 5 | disguise, insomnia, limber, magmaarmor, oblivious, owntempo, thermalexchange, trace, vitalspirit, waterbubble |
| onModifyAtk | 9 | 5 | blaze, firemane, hugepower, hustle, overgrow, purepower, swarm, torrent, waterbubble |
| onModifyMove | 9 | 8 | illuminate, infiltrator, keeneye, moldbreaker, scrappy, sheerforce, skilllink, stalwart, stancechange |
| onImmunity | 8 | 2 | icebody, magmaarmor, oblivious, overcoat, sandforce, sandrush, sandveil, snowcloak |
| onModifySpA | 7 | 4 | blaze, firemane, overgrow, solarpower, swarm, torrent, waterbubble |
| onSetStatus | 7 | 3 | insomnia, leafguard, limber, purifyingsalt, thermalexchange, vitalspirit, waterbubble |
| onDamage | 6 | 6 | berserk, disguise, heatproof, magicguard, rockhead, sturdy |
| onEnd | 6 | 5 | cloudnine, flashfire, illusion, supremeoverlord, unburden, unnerve |
| onResidual | 6 | 6 | cudchew, harvest, healer, hungerswitch, moody, speedboost |
| onTryAddVolatile | 6 | 3 | innerfocus, insomnia, leafguard, owntempo, purifyingsalt, vitalspirit |
| onModifyType | 5 | 2 | aerilate, dragonize, liquidvoice, pixilate, refrigerate |
| onSourceModifyDamage | 5 | 4 | auraguard, filter, multiscale, punkrock, solidrock |
| onModifySpe | 4 | 3 | chlorophyll, sandrush, surgesurfer, swiftswim |
| onWeather | 4 | 4 | dryskin, icebody, raindish, solarpower |
| onSourceModifyAtk | 4 | 3 | heatproof, purifyingsalt, thickfat, waterbubble |
| onSourceModifySpA | 4 | 4 | heatproof, purifyingsalt, thickfat, waterbubble |
| onAllyTryAddVolatile | 3 | 3 | aromaveil, flowerveil, sweetveil |
| onSwitchIn | 3 | 3 | cloudnine, imposter, zerotohero |
| onModifyDef | 3 | 3 | furcoat, grasspelt, marvelscale |
| onPrepareHit | 3 | 3 | libero, parentalbond, protean |
| onAllyTryHitSide | 3 | 3 | magicbounce, sapsipper, soundproof |
| onSwitchOut | 3 | 3 | naturalcure, regenerator, zerotohero |
| onHit | 2 | 2 | angerpoint, owntempo |
| onFoeTryMove | 2 | 2 | armortail, queenlymajesty |
| onAfterMoveSecondary | 2 | 2 | berserk, pickpocket |
| onAfterEachBoost | 2 | 2 | competitive, defiant |
| onSourceModifyAccuracy | 2 | 2 | compoundeyes, hustle |
| onSourceAfterFaint | 2 | 2 | eelevate, moxie |
| onAllySetStatus | 2 | 2 | flowerveil, sweetveil |
| onModifyPriority | 2 | 2 | galewings, prankster |
| onModifyCritRatio | 2 | 2 | merciless, superluck |
| onHitProtect | 2 | 1 | piercingdrill, unseenfist |
| onModifyAccuracy | 2 | 2 | sandveil, snowcloak |
| onModifySTAB | 1 | 1 | adaptability |
| onTryEatItem | 1 | 1 | berserk |
| onChangeBoost | 1 | 1 | contrary |
| onEatItem | 1 | 1 | cudchew |
| onAnyTryMove | 1 | 1 | damp |
| onAnyDamage | 1 | 1 | damp |
| onCriticalHit | 1 | 1 | disguise |
| onEffectiveness | 1 | 1 | disguise |
| onSourceBasePower | 1 | 1 | dryskin |
| onEmergencyExit | 1 | 1 | emergencyexit |
| onAnyBasePower | 1 | 1 | fairyaura |
| onAllyTryBoost | 1 | 1 | flowerveil |
| onAnyModifyDamage | 1 | 1 | friendguard |
| onBeforeSwitchIn | 1 | 1 | illusion |
| onFaint | 1 | 1 | illusion |
| onAnyRedirectTarget | 1 | 1 | lightningrod |
| onAfterMoveSecondarySelf | 1 | 1 | magician |
| onWeatherModifyDamage | 1 | 1 | megasol |
| onAnyInvulnerability | 1 | 1 | noguard |
| onAnyAccuracy | 1 | 1 | noguard |
| onSourceModifySecondaries | 1 | 1 | parentalbond |
| onSourceDamagingHit | 1 | 1 | poisontouch |
| onDeductPP | 1 | 1 | pressure |
| onFractionalPriority | 1 | 1 | quickdraw |
| onFoeTrapPokemon | 1 | 1 | shadowtag |
| onFoeMaybeTrapPokemon | 1 | 1 | shadowtag |
| onModifyDamage | 1 | 1 | sniper |
| onFlinch | 1 | 1 | steadfast |
| onAllyBasePower | 1 | 1 | steelyspirit |
| onAfterSetStatus | 1 | 1 | synchronize |
| onAnyModifyBoost | 1 | 1 | unaware |
| onAfterUseItem | 1 | 1 | unburden |
| onTakeItem | 1 | 1 | unburden |
| onFoeTryEatItem | 1 | 1 | unnerve |

#### item (31 hook paths)

| hook | n | templates | entities |
|---|---|---|---|
| onTakeItem | 81 | 2 | abomasite, absolite, absolitez, aerodactylite, aggronite, alakazite, altarianite, ampharosite, audinite, banettite, barbaracite, baxcalibrite, beedrillite, blastoisinite, blazikenite, cameruptite, chandelurite, charizarditex, charizarditey, chesnaughtite, chimechite, clefablite, crabominite, delphoxite, dragalgite, dragoninite, drampanite, eelektrossite, emboarite, excadrite, falinksite, feraligite, floettite, froslassite, galladite, garchompite, garchompitez, gardevoirite, gengarite, glalitite, ... (+41) |
| onEat | 24 | 5 | aspearberry, babiriberry, chartiberry, cheriberry, chestoberry, chopleberry, cobaberry, colburberry, habanberry, kasibberry, kebiaberry, leppaberry, lumberry, occaberry, passhoberry, payapaberry, rawstberry, rindoberry, roseliberry, shucaberry, sitrusberry, tangaberry, wacanberry, yacheberry |
| onBasePower | 20 | 3 | blackbelt, blackglasses, charcoal, dragonfang, fairyfeather, hardstone, magnet, metalcoat, miracleseed, muscleband, mysticwater, nevermeltice, poisonbarb, sharpbeak, silkscarf, silverpowder, softsand, spelltag, twistedspoon, wiseglasses |
| onSourceModifyDamage | 17 | 1 | babiriberry, chartiberry, chopleberry, cobaberry, colburberry, habanberry, kasibberry, kebiaberry, occaberry, passhoberry, payapaberry, rindoberry, roseliberry, shucaberry, tangaberry, wacanberry, yacheberry |
| onStart | 8 | 5 | airballoon, choicescarf, electricseed, grassyseed, metronome, mistyseed, psychicseed, whiteherb |
| onUpdate | 8 | 5 | aspearberry, cheriberry, chestoberry, leppaberry, lumberry, mentalherb, rawstberry, sitrusberry |
| onTerrainChange | 4 | 1 | electricseed, grassyseed, mistyseed, psychicseed |
| onDamagingHit | 2 | 2 | airballoon, rockyhelmet |
| onModifyMove | 2 | 2 | choicescarf, kingsrock |
| onModifySpe | 2 | 2 | choicescarf, ironball |
| onAfterMoveSecondary | 2 | 2 | ejectbutton, redcard |
| onModifyDamage | 2 | 2 | expertbelt, lifeorb |
| onModifyCritRatio | 2 | 2 | leek, scopelens |
| onResidual | 2 | 2 | leftovers, whiteherb |
| onAfterMoveSecondarySelf | 2 | 2 | lifeorb, shellbell |
| onSourceModifyAccuracy | 2 | 2 | widelens, zoomlens |
| onAfterSubDamage | 1 | 1 | airballoon |
| onTryHeal | 1 | 1 | bigroot |
| onModifyAccuracy | 1 | 1 | brightpowder |
| onDamage | 1 | 1 | focussash |
| onEffectiveness | 1 | 1 | ironball |
| onModifyAtk | 1 | 1 | lightball |
| onModifySpA | 1 | 1 | lightball |
| onAfterSetStatus | 1 | 1 | lumberry |
| onSourceTryPrimaryHit | 1 | 1 | normalgem |
| onFractionalPriority | 1 | 1 | quickclaw |
| onTryEatItem | 1 | 1 | sitrusberry |
| onAnySwitchIn | 1 | 1 | whiteherb |
| onAnyAfterMega | 1 | 1 | whiteherb |
| onAnyAfterMove | 1 | 1 | whiteherb |
| onUse | 1 | 1 | whiteherb |

#### condition (54 hook paths)

| hook | n | templates | entities |
|---|---|---|---|
| onStart | 54 | 38 | allyswitch, aquaring, attract, banefulbunker, brn, charge, choicelock, confusion, counter, curse, destinybond, disable, dragoncheer, encore, endure, flashfire, focusenergy, focuspunch, followme, frz, futuremove, glaiverush, healblock, healreplacement, helpinghand, imprison, kingsshield, leechseed, lockedmove, metronome, mirrorcoat, mustrecharge, noretreat, octolock, par, partiallytrapped, protect, psn, ragepowder, roost, ... (+14) |
| onEnd | 19 | 12 | attract, charge, confusion, disable, encore, flashfire, futuremove, healblock, lockedmove, partiallytrapped, perishsong, saltcure, stockpile, substitute, taunt, throatchop, twoturnmove, wish, yawn |
| onBeforeMove | 16 | 16 | attract, chillyreception, choicelock, confusion, destinybond, disable, flinch, frz, glaiverush, gravity, healblock, mustrecharge, par, slp, taunt, throatchop |
| durationCallback | 16 | 6 | auroraveil, electricterrain, grassyterrain, gravity, healblock, lightscreen, mistyterrain, partiallytrapped, psychicterrain, raindance, reflect, sandstorm, snowscape, sunnyday, tailwind, trickroom |
| onResidual | 15 | 13 | aquaring, brn, curse, encore, futuremove, grassyterrain, leechseed, lockedmove, octolock, partiallytrapped, perishsong, psn, saltcure, tox, wish |
| onRestart | 11 | 11 | allyswitch, charge, futuremove, healblock, helpinghand, lockedmove, minimize, smackdown, stall, stockpile, wish |
| onSideStart | 10 | 4 | auroraveil, lightscreen, quickguard, reflect, spikes, stealthrock, stickyweb, tailwind, toxicspikes, wideguard |
| onFieldStart | 10 | 5 | electricterrain, grassyterrain, gravity, mistyterrain, psychicterrain, raindance, sandstorm, snowscape, sunnyday, trickroom |
| onFieldEnd | 10 | 1 | electricterrain, grassyterrain, gravity, mistyterrain, psychicterrain, raindance, sandstorm, snowscape, sunnyday, trickroom |
| onTryHit | 7 | 7 | banefulbunker, kingsshield, protect, psychicterrain, quickguard, spikyshield, wideguard |
| onBasePower | 7 | 6 | charge, electricterrain, gem, grassyterrain, helpinghand, mistyterrain, psychicterrain |
| onDisableMove | 7 | 5 | choicelock, disable, encore, gravity, healblock, taunt, throatchop |
| onSwitchIn | 7 | 7 | healingwish, healreplacement, spikes, stealthrock, stickyweb, tox, toxicspikes |
| onSideEnd | 4 | 1 | auroraveil, lightscreen, reflect, tailwind |
| onHit | 4 | 4 | banefulbunker, focuspunch, kingsshield, spikyshield |
| onModifyMove | 4 | 4 | frz, gravity, healblock, throatchop |
| onFieldResidual | 4 | 2 | raindance, sandstorm, snowscape, sunnyday |
| onAnyModifyDamage | 3 | 2 | auroraveil, lightscreen, reflect |
| onMoveAborted | 3 | 3 | charge, destinybond, twoturnmove |
| onDamagingHit | 3 | 2 | counter, frz, mirrorcoat |
| onTryAddVolatile | 3 | 3 | electricterrain, focuspunch, mistyterrain |
| onTrapPokemon | 3 | 3 | octolock, partiallytrapped, trapped |
| onModifySpe | 3 | 3 | par, tailwind, unburden |
| onUpdate | 2 | 2 | attract, fling |
| onAfterMove | 2 | 2 | charge, lockedmove |
| onRedirectTarget | 2 | 1 | counter, mirrorcoat |
| onModifyCritRatio | 2 | 2 | dragoncheer, focusenergy |
| onSetStatus | 2 | 2 | electricterrain, mistyterrain |
| onFoeRedirectTarget | 2 | 2 | followme, ragepowder |
| onAccuracy | 2 | 2 | glaiverush, minimize |
| onSourceModifyDamage | 2 | 2 | glaiverush, minimize |
| onLockMove | 2 | 2 | lockedmove, twoturnmove |
| onWeatherModifyDamage | 2 | 2 | raindance, sunnyday |
| onSideRestart | 2 | 1 | spikes, toxicspikes |
| onFaint | 1 | 1 | destinybond |
| onDamage | 1 | 1 | endure |
| onModifyAtk | 1 | 1 | flashfire |
| onModifySpA | 1 | 1 | flashfire |
| onAfterMoveSecondary | 1 | 1 | frz |
| onModifyAccuracy | 1 | 1 | gravity |
| onTryHeal | 1 | 1 | healblock |
| onSwap | 1 | 1 | healingwish |
| onFoeDisableMove | 1 | 1 | imprison |
| onFoeBeforeMove | 1 | 1 | imprison |
| onTryMove | 1 | 1 | metronome |
| onModifyDamage | 1 | 1 | metronome |
| onType | 1 | 1 | roost |
| onModifySpD | 1 | 1 | sandstorm |
| onWeather | 1 | 1 | sandstorm |
| onModifyDef | 1 | 1 | snowscape |
| onStallMove | 1 | 1 | stall |
| onTryPrimaryHit | 1 | 1 | substitute |
| onImmunity | 1 | 1 | sunnyday |
| onFieldRestart | 1 | 1 | trickroom |

### T6. v1 data-only field frequency


**move**: `type` 454, `target` 454, `accuracy` 454, `critRatio` 454, `category` 454, `pp` 454, `flags` 450, `basePower` 290, `secondaries` 181, `secondary` 176, `ignoreImmunity` 145, `self` 64, `boosts` 45, `volatileStatus` 39, `priority` 38, `multihit` 14, `sideCondition` 10, `recoil` 9, `selfSwitch` 8, `drain` 8, `selfBoost` 7, `stallingMove` 6, `selfdestruct` 6, `weather` 5, `heal` 5, `forceSwitch` 4, `terrain` 4, `ohko` 4, `hasCrashDamage` 3, `willCrit` 3, `slotCondition` 3, `thawsTarget` 3, `callsMove` 2, `ignoreDefensive` 2, `ignoreEvasion` 2, `breaksProtect` 2, `pseudoWeather` 2, `damage` 2, `multiaccuracy` 2, `sleepUsable` 2, `overrideOffensiveStat` 1, `smartTarget` 1, `hasSheerForceBoost` 1, `overrideOffensivePokemon` 1, `tags` 1, `overrideDefensiveStat` 1, `noPPBoosts` 1, `tracksTarget` 1, `mindBlownRecoil` 1

**ability**: `flags` 63, `onBasePowerPriority` 15, `onModifyAtkPriority` 8, `onModifySpAPriority` 6, `onResidualOrder` 6, `onModifyTypePriority` 5, `onResidualSubOrder` 5, `onSourceModifyAtkPriority` 4, `onSourceModifySpAPriority` 4, `onDamagingHitOrder` 3, `onModifyDefPriority` 3, `onTryHitPriority` 3, `onModifyMovePriority` 3, `onSourceModifyAccuracyPriority` 2, `onDamagePriority` 2, `onSwitchInPriority` 2, `onModifyAccuracyPriority` 2, `suppressWeather` 1, `onSourceBasePowerPriority` 1, `onAnyBasePowerPriority` 1, `onWeatherModifyDamagePriority` 1, `onAnyInvulnerabilityPriority` 1, `onFractionalPriorityPriority` 1, `onFractionalPriority` 1, `onAllyBasePowerPriority` 1

**item**: `fling` 159, `itemUser` 83, `megaStone` 81, `isBerry` 24, `naturalGift` 24, `onBasePowerPriority` 20, `onSwitchInPriority` 4, `boosts` 4, `onResidualOrder` 2, `onSourceModifyAccuracyPriority` 2, `onTryHealPriority` 1, `onModifyAccuracyPriority` 1, `isChoice` 1, `onAfterMoveSecondaryPriority` 1, `onDamagePriority` 1, `onModifyMovePriority` 1, `onResidualSubOrder` 1, `onModifyAtkPriority` 1, `onModifySpAPriority` 1, `onAfterSetStatusPriority` 1, `isGem` 1, `onFractionalPriorityPriority` 1, `onDamagingHitOrder` 1, `onAfterMoveSecondarySelfPriority` 1, `onAnySwitchInPriority` 1

### T7. v1 CUSTOM entities ranked by complexity score (= LoC + 6*fns + 4*core-code refs + 3*mentions by other entities)

| # | kind | id | score | fns | LoC | core refs | mentioned by | flag | hooks |
|---|---|---|---|---|---|---|---|---|---|
| 1 | condition | sunnyday | 190 | 6 | 40 | 3 | 34 |  | durationCallback, onFieldEnd, onFieldResidual, onFieldStart, onImmunity, onWeatherModifyDamage |
| 2 | move | substitute | 189 | 5 | 59 | 7 | 24 | Substitute | onEnd, onHit, onStart, onTryHit, onTryPrimaryHit |
| 3 | condition | sandstorm | 165 | 6 | 29 | 1 | 32 |  | durationCallback, onFieldEnd, onFieldResidual, onFieldStart, onModifySpD, onWeather |
| 4 | condition | substitute | 164 | 3 | 46 | 7 | 24 | Substitute | onEnd, onStart, onTryPrimaryHit |
| 5 | condition | raindance | 156 | 5 | 32 | 1 | 30 |  | durationCallback, onFieldEnd, onFieldResidual, onFieldStart, onWeatherModifyDamage |
| 6 | condition | snowscape | 134 | 5 | 26 | 0 | 26 |  | durationCallback, onFieldEnd, onFieldResidual, onFieldStart, onModifyDef |
| 7 | condition | gravity | 127 | 7 | 66 | 1 | 5 | field rule changes | durationCallback, onBeforeMove, onDisableMove, onFieldEnd, onFieldStart, onModifyAccuracy, onModifyMove |
| 8 | condition | slp | 126 | 2 | 29 | 7 | 19 |  | onBeforeMove, onStart |
| 9 | condition | lockedmove | 119 | 6 | 28 | 4 | 13 |  | onAfterMove, onEnd, onLockMove, onResidual, onRestart, onStart |
| 10 | condition | confusion | 117 | 3 | 31 | 2 | 20 |  | onBeforeMove, onEnd, onStart |
| 11 | condition | healblock | 113 | 8 | 52 | 1 | 3 | move lock | durationCallback, onBeforeMove, onDisableMove, onEnd, onModifyMove, onRestart, onStart, onTryHeal |
| 12 | condition | partiallytrapped | 103 | 5 | 25 | 9 | 4 |  | durationCallback, onEnd, onResidual, onStart, onTrapPokemon |
| 13 | condition | frz | 99 | 5 | 38 | 4 | 5 |  | onAfterMoveSecondary, onBeforeMove, onDamagingHit, onModifyMove, onStart |
| 14 | condition | futuremove | 99 | 4 | 39 | 3 | 8 |  | onEnd, onResidual, onRestart, onStart |
| 15 | condition | twoturnmove | 99 | 4 | 23 | 1 | 16 |  | onEnd, onLockMove, onMoveAborted, onStart |
| 16 | condition | electricterrain | 91 | 6 | 37 | 0 | 6 |  | durationCallback, onBasePower, onFieldEnd, onFieldStart, onSetStatus, onTryAddVolatile |
| 17 | condition | stall | 86 | 3 | 16 | 1 | 16 | turn order | onRestart, onStallMove, onStart |
| 18 | condition | grassyterrain | 85 | 5 | 34 | 0 | 7 |  | durationCallback, onBasePower, onFieldEnd, onFieldStart, onResidual |
| 19 | condition | mistyterrain | 84 | 6 | 36 | 0 | 4 |  | durationCallback, onBasePower, onFieldEnd, onFieldStart, onSetStatus, onTryAddVolatile |
| 20 | move | disable | 83 | 5 | 44 | 0 | 3 | move lock | onBeforeMove, onDisableMove, onEnd, onStart, onTryHit |
| 21 | condition | psychicterrain | 82 | 5 | 37 | 0 | 5 | priority block | durationCallback, onBasePower, onFieldEnd, onFieldStart, onTryHit |
| 22 | condition | charge | 81 | 6 | 33 | 0 | 4 |  | onAfterMove, onBasePower, onEnd, onMoveAborted, onRestart, onStart |
| 23 | move | curse | 79 | 5 | 34 | 3 | 1 |  | onHit, onModifyMove, onResidual, onStart, onTryHit |
| 24 | item | metronome | 76 | 4 | 32 | 5 | 0 | Metronome | onModifyDamage, onStart, onTryMove |
| 25 | condition | psn | 76 | 2 | 10 | 3 | 14 |  | onResidual, onStart |
| 26 | move | auroraveil | 75 | 5 | 27 | 0 | 6 |  | durationCallback, onAnyModifyDamage, onSideEnd, onSideStart, onTry |
| 27 | condition | tox | 73 | 3 | 19 | 3 | 8 |  | onResidual, onStart, onSwitchIn |
| 28 | condition | disable | 72 | 4 | 39 | 0 | 3 | move lock | onBeforeMove, onDisableMove, onEnd, onStart |
| 29 | move | throatchop | 70 | 6 | 28 | 0 | 2 | move lock | onBeforeMove, onDisableMove, onEnd, onHit, onModifyMove, onStart |
| 30 | ability | waterbubble | 69 | 6 | 33 | 0 | 0 |  | onModifyAtk, onModifySpA, onSetStatus, onSourceModifyAtk, onSourceModifySpA, onUpdate |
| 31 | condition | attract | 69 | 4 | 33 | 0 | 4 |  | onBeforeMove, onEnd, onStart, onUpdate |
| 32 | condition | brn | 69 | 2 | 12 | 3 | 11 |  | onResidual, onStart |
| 33 | condition | encore | 69 | 4 | 39 | 0 | 2 | Encore | onDisableMove, onEnd, onResidual, onStart |
| 34 | move | shedtail | 68 | 3 | 20 | 6 | 2 | Shed Tail | onHit, onTryHit |
| 35 | move | weatherball | 67 | 2 | 39 | 1 | 4 |  | onModifyMove, onModifyType |
| 36 | condition | metronome | 67 | 3 | 29 | 5 | 0 | Metronome | onModifyDamage, onStart, onTryMove |
| 37 | move | yawn | 66 | 3 | 12 | 0 | 12 |  | onEnd, onStart, onTryHit |
| 38 | ability | disguise | 66 | 4 | 34 | 2 | 0 | Disguise (form change) | onCriticalHit, onDamage, onEffectiveness, onUpdate |
| 39 | ability | flashfire | 66 | 6 | 30 | 0 | 0 |  | onEnd, onModifyAtk, onModifySpA, onStart, onTryHit |
| 40 | condition | auroraveil | 66 | 4 | 24 | 0 | 6 |  | durationCallback, onAnyModifyDamage, onSideEnd, onSideStart |
| 41 | move | counter | 65 | 6 | 26 | 0 | 1 |  | beforeTurnCallback, damageCallback, onDamagingHit, onRedirectTarget, onStart, onTry |
| 42 | move | mirrorcoat | 65 | 6 | 26 | 0 | 1 |  | beforeTurnCallback, damageCallback, onDamagingHit, onRedirectTarget, onStart, onTry |
| 43 | move | stockpile | 65 | 4 | 35 | 0 | 2 |  | onEnd, onRestart, onStart, onTry |
| 44 | item | whiteherb | 65 | 6 | 29 | 0 | 0 |  | onAnyAfterMega, onAnyAfterMove, onAnySwitchIn, onResidual, onStart, onUse |
| 45 | move | revivalblessing | 63 | 1 | 5 | 13 | 0 | Revival Blessing | onTryHit |
| 46 | condition | smackdown | 61 | 2 | 27 | 1 | 6 |  | onRestart, onStart |
| 47 | condition | throatchop | 61 | 5 | 25 | 0 | 2 | move lock | onBeforeMove, onDisableMove, onEnd, onModifyMove, onStart |
| 48 | condition | par | 60 | 3 | 20 | 1 | 6 |  | onBeforeMove, onModifySpe, onStart |
| 49 | condition | taunt | 60 | 4 | 23 | 1 | 3 | move lock | onBeforeMove, onDisableMove, onEnd, onStart |
| 50 | condition | toxicspikes | 60 | 3 | 21 | 0 | 7 |  | onSideRestart, onSideStart, onSwitchIn |
| 51 | condition | trickroom | 60 | 4 | 20 | 4 | 0 | turn-order inversion | durationCallback, onFieldEnd, onFieldRestart, onFieldStart |
| 52 | move | allyswitch | 57 | 4 | 33 | 0 | 0 | Ally Switch | onHit, onPrepareHit, onRestart, onStart |
| 53 | ability | illusion | 56 | 4 | 32 | 0 | 0 | Illusion | onBeforeSwitchIn, onDamagingHit, onEnd, onFaint |
| 54 | condition | stockpile | 56 | 3 | 32 | 0 | 2 |  | onEnd, onRestart, onStart |
| 55 | move | destinybond | 55 | 5 | 25 | 0 | 0 |  | onBeforeMove, onFaint, onMoveAborted, onPrepareHit, onStart |
| 56 | move | leechseed | 55 | 3 | 17 | 2 | 4 |  | onResidual, onStart, onTryImmunity |
| 57 | condition | kingsshield | 55 | 3 | 26 | 2 | 1 |  | onHit, onStart, onTryHit |
| 58 | condition | yawn | 55 | 2 | 7 | 0 | 12 |  | onEnd, onStart |
| 59 | move | fling | 54 | 2 | 39 | 0 | 1 | item consumption | onPrepareHit, onUpdate |
| 60 | move | focuspunch | 53 | 5 | 20 | 0 | 1 | priority charge | beforeMoveCallback, onHit, onStart, onTryAddVolatile, priorityChargeCallback |

### T8. Notorious mechanics: legal in Reg M-C? in v1?

| id | legal as | in v1 | note |
|---|---|---|---|
| commander | not legal | - | Commander (Tatsugiri/Dondozo) |
| allyswitch | move,condition | yes | Ally Switch |
| protosynthesis | not legal | - | Paradox |
| quarkdrive | not legal | - | Paradox |
| illusion | ability | yes | Illusion |
| transform | move | yes | Transform |
| imposter | ability | yes | Imposter |
| dancer | not legal | - | Dancer |
| instruct | move | yes | Instruct |
| encore | move,condition | yes | Encore |
| disguise | ability | yes | Disguise (form change) |
| stancechange | ability | yes | Stance Change (form change) |
| zerotohero | ability | yes | Zero to Hero (form change) |
| hungerswitch | ability | yes | Hunger Switch (form change) |
| forecast | ability | no | Forecast (form change) |
| trace | ability | yes | Trace |
| skillswap | move | yes | Skill Swap |
| roleplay | move | no | Role Play |
| magicbounce | ability | yes | Magic Bounce |
| neutralizinggas | not legal | - | Neutralizing Gas |
| substitute | move,condition | yes | Substitute |
| batonpass | move | yes | Baton Pass |
| shedtail | move | yes | Shed Tail |
| revivalblessing | move,condition | yes | Revival Blessing |
| afteryou | move | yes | After You |
| quash | move | yes | Quash |
| followme | move,condition | yes | redirection |
| ragepowder | move,condition | yes | redirection |
| lightningrod | ability | yes | redirection |
| stormdrain | not legal | - | redirection |
| wideguard | move,condition | yes | spread block |
| quickguard | move,condition | yes | priority block |
| fakeout | move | yes |  |
| trickroom | move,condition | yes | turn-order inversion |
| tailwind | move,condition | yes |  |
| gravity | move,condition | yes | field rule changes |
| perishsong | move,condition | yes | perish counter |
| unburden | ability,condition | yes | item-loss trigger |
| dragonize | ability | yes | Champions mega ability |
| megasol | ability | yes | Champions mega ability |
| piercingdrill | ability | yes | Champions mega ability |
| spicyspray | ability | yes | Champions mega ability |
| firemane | ability | yes | Champions mega ability |
| eelevate | ability | yes | Champions mega ability |
| auraguard | ability | yes | Champions mega ability |
