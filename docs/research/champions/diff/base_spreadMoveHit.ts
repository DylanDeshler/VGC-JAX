spreadMoveHit(
	targets: SpreadMoveTargets, pokemon: Pokemon, moveOrMoveName: ActiveMove,
	hitEffect?: Dex.HitEffect, isSecondary?: boolean, isSelf?: boolean
): [SpreadMoveDamage, SpreadMoveTargets] {
	// Hardcoded for single-target purposes
	// (no spread moves have any kind of onTryHit handler)
	const target = targets[0];
	let damage: (number | boolean | undefined)[] = [];
	for (const i of targets.keys()) {
		damage[i] = true;
	}
	const move = this.dex.getActiveMove(moveOrMoveName);
	let hitResult: boolean | number | null = true;
	let moveData = hitEffect as ActiveMove;
	if (!moveData) moveData = move;
	if (!moveData.flags) moveData.flags = {};
	if (move.target === 'all' && !isSelf) {
		hitResult = this.battle.singleEvent('TryHitField', moveData, {}, target || null, pokemon, move);
	} else if ((move.target === 'foeSide' || move.target === 'allySide' || move.target === 'allyTeam') && !isSelf) {
		hitResult = this.battle.singleEvent('TryHitSide', moveData, {}, target || null, pokemon, move);
	} else if (target) {
		hitResult = this.battle.singleEvent('TryHit', moveData, {}, target, pokemon, move);
	}
	if (!hitResult) {
		if (hitResult === false) {
			this.battle.add('-fail', pokemon);
			this.battle.attrLastMove('[still]');
		}
		return [[false], targets]; // single-target only
	}

	// 0. check for substitute
	if (!isSecondary && !isSelf) {
		if (move.target !== 'all' && move.target !== 'allyTeam' && move.target !== 'allySide' && move.target !== 'foeSide') {
			damage = this.tryPrimaryHitEvent(damage, targets, pokemon, move, moveData, isSecondary);
		}
	}

	for (const i of targets.keys()) {
		if (damage[i] === this.battle.HIT_SUBSTITUTE) {
			damage[i] = true;
			targets[i] = null;
		}
		if (targets[i] && isSecondary && !moveData.self) {
			damage[i] = true;
		}
		if (!damage[i]) targets[i] = false;
	}
	// 1. call to this.battle.getDamage
	damage = this.getSpreadDamage(damage, targets, pokemon, move, moveData, isSecondary, isSelf);

	for (const i of targets.keys()) {
		if (damage[i] === false) targets[i] = false;
	}

	// 2. call to this.battle.spreadDamage
	damage = this.battle.spreadDamage(damage, targets, pokemon, move);

	for (const i of targets.keys()) {
		if (damage[i] === false) targets[i] = false;
	}

	// 3. onHit event happens here
	damage = this.runMoveEffects(damage, targets, pokemon, move, moveData, isSecondary, isSelf);

	for (const i of targets.keys()) {
		if (!damage[i] && damage[i] !== 0) targets[i] = false;
	}

	// steps 4 and 5 can mess with this.battle.activeTarget, which needs to be preserved for Dancer
	const activeTarget = this.battle.activeTarget;

	// 4. self drops (start checking for targets[i] === false here)
	if (moveData.self && !move.selfDropped) this.selfDrops(targets, pokemon, move, moveData, isSecondary);

	// 5. secondary effects
	if (moveData.secondaries) this.secondaries(targets, pokemon, move, moveData, isSelf);

	this.battle.activeTarget = activeTarget;

	// 6. force switch
	if (moveData.forceSwitch) damage = this.forceSwitch(damage, targets, pokemon, move);

	for (const i of targets.keys()) {
		if (!damage[i] && damage[i] !== 0) targets[i] = false;
	}

	const damagedTargets: Pokemon[] = [];
	const damagedDamage = [];
	for (const [i, t] of targets.entries()) {
		if (typeof damage[i] === 'number' && t) {
			damagedTargets.push(t);
			damagedDamage.push(damage[i]);
		}
	}
	const pokemonOriginalHP = pokemon.hp;
	if (damagedDamage.length && !isSecondary && !isSelf) {
		if (this.battle.gen >= 5) {
			this.battle.runEvent('DamagingHit', damagedTargets, pokemon, move, damagedDamage);
		}
		if (moveData.onAfterHit && pokemon.hp) {
			for (const t of damagedTargets) {
				this.battle.singleEvent('AfterHit', moveData, {}, t, pokemon, move);
			}
		}
		if (this.battle.gen < 5) {
			this.battle.runEvent('DamagingHit', damagedTargets, pokemon, move, damagedDamage);
		}
		this.battle.runEvent('EmergencyExit', pokemon, undefined, undefined, pokemonOriginalHP);
	}

	return [damage, targets];
}
