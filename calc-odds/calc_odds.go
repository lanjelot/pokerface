package main

import (
	"fmt"
	"os"

	"github.com/chehsunliu/poker"
	"gonum.org/v1/gonum/stat/combin"
)

var (
	strRanks = "23456789TJQKA"
	strSuits = "shdc"
)

type Card = poker.Card

func makeDeck(exclude []Card) []Card {
	var cards []Card

	for _, rank := range strRanks {
		for _, suit := range strSuits {
			card := poker.NewCard(string(rank)+string(suit))
			if containsCard(exclude, card) == false {
				cards = append(cards, card)
			}				
		}
	}

	return cards
}

func containsCard(cards []Card, card Card) bool {
	for _, v := range cards {
		if v == card {
			return true
		}
	}
	return false
	
}

func removeCards(deck []Card, cards []Card) []Card {
	new := make([]Card, len(deck) - len(cards))
	i := 0
	for _, v := range deck {
		if containsCard(cards, v) == false {
			new[i] = v
			i += 1			
		}
	}
	return new
}

func main() {

	argc := len(os.Args) - 1

	if argc < 5 {
		fmt.Println("Usage:")
		fmt.Println("$ calc-odds Ah Kh 2h 3c 4s       # flop")
		fmt.Println("$ calc-odds Ah Kh 2h 3c 4s Th    # turn")
		fmt.Println("$ calc-odds Ah Kh 2h 3c 4s Th 5c # river")
		os.Exit(2)
	}
	
	h1 := poker.NewCard(os.Args[1])
	h2 := poker.NewCard(os.Args[2])
	b1 := poker.NewCard(os.Args[3])
	b2 := poker.NewCard(os.Args[4])
	b3 := poker.NewCard(os.Args[5])

	losses, wins := 0, 0

	if argc == 5 {
		losses, wins = flop(h1, h2, b1, b2, b3)
	}
	if argc == 6 {
		b4 := poker.NewCard(os.Args[6])
		losses, wins = turn(h1, h2, b1, b2, b3, b4)
	}
	if argc == 7 {
		b4 := poker.NewCard(os.Args[6])
		b5 := poker.NewCard(os.Args[7])
		losses, wins = river(h1, h2, b1, b2, b3, b4, b5)
	}

	fmt.Println(losses, wins)	
}

func flop(h1 Card, h2 Card, b1 Card, b2 Card, b3 Card) (int, int) {
	losses := 0
	wins := 0

	exclude := []Card {h1, h2, b1, b2, b3}
	deck := makeDeck(exclude)
	list := combin.Combinations(len(deck), 2)	
	for _, v := range list {
		b4 := deck[v[0]]
		b5 := deck[v[1]]

		x, y := river(h1, h2, b1, b2, b3, b4, b5)
		
		losses += x
		wins += y
	}
	
	return losses, wins
}


func turn(h1 Card, h2 Card, b1 Card, b2 Card, b3 Card, b4 Card) (int, int) {
	losses := 0
	wins := 0

	exclude := []Card {h1, h2, b1, b2, b3, b4}
	deck := makeDeck(exclude)
	for _, b5 := range deck {
		x, y := river(h1, h2, b1, b2, b3, b4, b5)

		losses += x
		wins += y
	}
			
	return losses, wins
}


func river(h1 Card, h2 Card, b1 Card, b2 Card, b3 Card, b4 Card, b5 Card) (int, int) {
	losses := 0
	wins := 0

	my_hand := []Card {h1, h2, b1, b2, b3, b4, b5}
	my_score := poker.Evaluate(my_hand)

	deck := makeDeck(my_hand)
	list := combin.Combinations(len(deck), 2)
	for _, v := range list {
		th1 := deck[v[0]]
		th2 := deck[v[1]]

		their_hand := []Card {th1, th2, b1, b2, b3, b4, b5}
		their_score := poker.Evaluate(their_hand)
		if my_score > their_score {
			losses += 1
		} else {
			wins += 1 // wins or ties
		}
	}
	
	return losses, wins
}