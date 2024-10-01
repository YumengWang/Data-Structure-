// Project UID 1d9f47bfc76643019cfbf037641defe1

#include <cassert>
#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <array>
#include <vector>
#include <sstream>
#include <algorithm>
#include "Player.h"

class SimplePlayer : public Player{
public:
    SimplePlayer(const std::string &name): name(name) {
    }
    
    //EFFECTS returns player's name
    virtual const std::string & get_name() const{
        return name;
    }

    //REQUIRES player has less than MAX_HAND_SIZE cards
    //EFFECTS  adds Card c to Player's hand
    virtual void add_card(const Card &c) {
        if (hands.size() < MAX_HAND_SIZE) {
            hands.push_back(c);
        }
    }
    
    
    //REQUIRES round is 1 or 2
    //MODIFIES order_up_suit
    //EFFECTS If Player wishes to order up a trump suit then return true and
    //  change order_up_suit to desired suit.  If Player wishes to pass, then do
    //  not modify order_up_suit and return false.
    virtual bool make_trump(const Card &upcard, bool is_dealer,
                            int round, std::string &order_up_suit) const{
        int trump_face_cards = 0;
        int next_face_cards = 0;
        if (round == 1) {
            for (int i = 0; i < hands.size(); ++i) {
                if (hands[i].is_trump(upcard.get_suit()) &&
                    hands[i].is_face()) {
                    trump_face_cards++;
                }
            }
            if (trump_face_cards >= 2) {
                order_up_suit = upcard.get_suit();
                return true;
            }
        }
        if (round == 2) {
            if (is_dealer) {
                order_up_suit = Suit_next(upcard.get_suit());
                return true;
            }
            for (int i = 0; i < hands.size(); ++i) {
                if (hands[i].is_trump(Suit_next(upcard.get_suit())) &&
                    hands[i].is_face()) {
                    next_face_cards++;
                }
            }
            if (next_face_cards >= 1) {
                order_up_suit = Suit_next(upcard.get_suit());
                return true;
            }
        }
        return false;
    }

    //REQUIRES Player has at least one card
    //EFFECTS  Player adds one card to hand and removes one card from hand.
    virtual void add_and_discard(const Card &upcard) {
        hands.push_back(upcard);
        int smallest_card_order = 0;
        for (int i = 0; i < hands.size(); ++i) {
            if (Card_less(hands[i], hands[smallest_card_order],
                          upcard.get_suit())) {
                smallest_card_order = i;
            }
        }
        hands.erase(hands.begin() + smallest_card_order);
    }

    //REQUIRES Player has at least one card, trump is a valid suit
    //EFFECTS  Leads one Card from Player's hand according to their strategy
    //  "Lead" means to play the first Card in a trick.  The card
    //  is removed the player's hand.
    virtual Card lead_card(const std::string &trump) {
        std::vector<int> nontrump_card;
        Card lead_card = hands[0];
        int lead_card_number = 0;
        for (int i = 0; i < hands.size(); ++i) {
            if (!hands[i].is_trump(trump)) {
                nontrump_card.push_back(i);
            }
        }
        if (nontrump_card.size() != 0) {
            lead_card = hands[nontrump_card[0]];
            lead_card_number = nontrump_card[0];
            for (int i = 0; i < nontrump_card.size(); ++i) {
                if (Card_less(lead_card, hands[nontrump_card[i]], trump)) {
                    lead_card = hands[nontrump_card[i]];
                    lead_card_number = nontrump_card[i];
              }
            }
            hands.erase(hands.begin() + lead_card_number);
            return lead_card;
        }
        lead_card = hands[0];
        for (int i = 0; i < hands.size(); ++i) {
            if (Card_less(lead_card, hands[i], trump)) {
                lead_card = hands[i];
                lead_card_number = i;
          }
        }
        hands.erase(hands.begin() + lead_card_number);
        return lead_card;
    }

    //REQUIRES Player has at least one card, trump is a valid suit
    //EFFECTS  Plays one Card from Player's hand according to their strategy.
    //  The card is removed from the player's hand.
    virtual Card play_card(const Card &led_card, const std::string &trump) {
        std::vector<int> suit_number;
        std::string led_suit = led_card.get_suit();
        Card play_card = hands[0];
        for (int i = 0; i < hands.size(); i++) {
            if (hands[i].get_suit(trump) == led_suit) {
                suit_number.push_back(i);
            }
        }
        if (suit_number.size() > 0) {
            int highest_number = suit_number[0];
            for (int i = 0; i < suit_number.size(); ++i) {
                if (Card_less(hands[highest_number],
                              hands[suit_number[i]], led_card, trump)) {
                    highest_number = suit_number[i];
                }
            }
            play_card = hands[highest_number];
            hands.erase(hands.begin() + highest_number);
            return play_card;
        }
        int lowest_number = 0;
        for (int i = 0; i < hands.size(); ++i) {
            if (Card_less(hands[i], hands[lowest_number], led_card, trump)) {
                lowest_number = i;
            }
        }
        play_card = hands[lowest_number];
        hands.erase(hands.begin() + lowest_number);
        return play_card;
    }
    
    // Needed to avoid some compiler errors
    virtual ~SimplePlayer() {}
    
    
private:
    std::string name;
    std::vector<Card> hands;
};

class HumanPlayer : public Player{
public:
    HumanPlayer(const std::string &name): name(name) {
    }
    
    //EFFECTS returns player's name
    virtual const std::string & get_name() const {
        return name;
    }

    //REQUIRES player has less than MAX_HAND_SIZE cards
    //EFFECTS  adds Card c to Player's hand
    virtual void add_card(const Card &c) {
        if (hands.size() < MAX_HAND_SIZE) {
            hands.push_back(c);
        }
    }
    
    //REQUIRES round is 1 or 2
    //MODIFIES order_up_suit
    //EFFECTS If Player wishes to order up a trump suit then return true and
    //  change order_up_suit to desired suit.  If Player wishes to pass, then do
    //  not modify order_up_suit and return false.
    virtual bool make_trump(const Card &upcard, bool is_dealer,
                            int round, std::string &order_up_suit) const {
        std::vector<int> hands_card;
        //std::vector<int> ordered_card;
        //int card_number = 0;
        std::string decision = "";
        for (int i = 0; i < hands.size(); ++i) {
            hands_card.push_back(i);
        }
        std::sort(hands_card.begin(), hands_card.end());
        /*for (int m = 0; m < hands.size(); ++m) {
            for (int j = 0; j < hands_card.size(); ++j) {
                for (int k = 0; k < hands_card.size(); ++k) {
                    if (hands[j] > hands[k]) {
                        break;
                    }
                    if (k == hands_card.size() - 1) {
                        card_number = j;
                    }
                }
            }
            hands_card.erase(hands_card.begin() + card_number);
            ordered_card.push_back(card_number);
        }*/
        for (int i = 0; i < hands_card.size(); ++i) {
            std::cout << "Human player " << name << "'s hand: [" << i <<
            "] " << hands[i] << std::endl;
        }
        std::cout << "Human player " << name
        << ", please enter a suit, or \"pass\":"<< std::endl;
        std::cin >> decision;
        if (decision == "pass") {
            std::cout << name << " passes" << std::endl;
        }
        else {
            std::cout << name << " orders up " << decision << std::endl;
        }
        return false;
    }
    
    //REQUIRES Player has at least one card
    //EFFECTS  Player adds one card to hand and removes one card from hand.
    virtual void add_and_discard(const Card &upcard) {
        int discard_number = 0;
        std::sort(hands.begin(), hands.end());
        for (int i = 0; i < hands.size(); ++i) {
            std::cout << "Human player " << name << "'s hand: [" << i <<
            "] " << hands[i] << std::endl;
        }
        hands.push_back(upcard);
        std::cout << "Discard upcard: [-1]" << std::endl;
        std::cout << "Human player " << name
        << ", please select a card to discard:"<< std::endl;
        std::cin >> discard_number;
        hands.erase(hands.begin() + discard_number);
        return;
    }
    
    
    //REQUIRES Player has at least one card, trump is a valid suit
    //EFFECTS  Leads one Card from Player's hand according to their strategy
    //  "Lead" means to play the first Card in a trick.  The card
    //  is removed the player's hand.
    virtual Card lead_card(const std::string &trump) {
        std::vector<Card> hand_card;
        int index = 0;
        for (int i = 0; i < hands.size(); ++i) {
            hand_card.push_back(hands[i]);
        }
        std::sort(hand_card.begin(), hand_card.end());
        for (int i = 0; i < hand_card.size(); ++i) {
            std::cout << "Human player " << name << "'s hand: [" << i <<
            "] " << hand_card[i] << std::endl;
        }
        std::cout << "Human player " << name
        << ", please select a card:"<< std::endl;
        std::cin >> index;
        Card temp_card = hand_card[index];
        int hands_number = 0;
        for (int i = 0; i < hands.size(); ++i) {
            if (hands[i] == hand_card[index]) {
                hands_number = i;
            }
        }
        hands.erase(hands.begin() + hands_number);
        return temp_card;
    }
    
    //REQUIRES Player has at least one card, trump is a valid suit
    //EFFECTS  Plays one Card from Player's hand according to their strategy.
    //  The card is removed from the player's hand.
    virtual Card play_card(const Card &led_card, const std::string &trump) {
        std::vector<Card> hands_card;
        int number = 0;
        for (int i = 0; i < hands.size(); ++i) {
            hands_card.push_back(hands[i]);
        }
        std::sort(hands_card.begin(), hands_card.end());
        for (int i = 0; i < hands_card.size(); ++i) {
            std::cout << "Human player " << name << "'s hand: [" << i <<
            "] " << hands_card[i] << std::endl;
        }
        std::cout << "Human player " << name
        << ", please select a card:"<< std::endl;
        std::cin >> number;
        Card temp = hands_card[number];
        int hands_number = 0;
        for (int i = 0; i < hands.size(); ++i) {
            if (hands[i] == hands_card[number]) {
                hands_number = i;
            }
        }
        hands.erase(hands.begin() + hands_number);
        return temp;
    }

    
    
    // Needed to avoid some compiler errors
    virtual ~HumanPlayer() {}
    
private:
    std::string name;
    std::vector<Card> hands;
    bool is_dealer;
};

Player * Player_factory(const std::string &name,
                        const std::string &strategy) {
    if (strategy == "Simple") {
        return new SimplePlayer(name);
    }
    if (strategy == "Human") {
        return new HumanPlayer(name);
    }
    assert(false);
    return nullptr;
}

//EFFECTS: Prints player's name to os
std::ostream & operator<<(std::ostream &os, const Player &p) {
    os << p.get_name();
    return os;
}
