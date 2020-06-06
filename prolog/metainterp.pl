:- op(1100,fx,query).
:- op(1060,xfx,if).

query Query :-
  query(Query,inf).

query(Query,B) :-
  B > 0,
  bound_init(Query,B),
  step(Query).

step(true) :- !.
step(Query) :-
  (Query if Body),
  functor(Query,Pred,Arity),
  bound(Pred,Arity,_),
  step_act(Body).

step_act((B1,B2)) :-
  !, step(B1), step(B2).
step_act(B) :- step(B).

% this version keeps an overall depth bound
% initialization
:- dynamic qb/2.
bound_init(_Q,B) :-
  retractall(qb(_,_)),
  assert(qb(0,B)).

% forward step
bound(_Pred,_Arity,NewN) :-
  retract(qb(N,B)),
  NewN is N + 1, NewN < B,
  assert(qb(NewN,B)).

% backward step - make sure this fails
bound(_Pred,_Arity,_) :-
  retract(qb(NewN,B)),
  N is NewN - 1,
  assert(qb(N,B)),
  fail.

