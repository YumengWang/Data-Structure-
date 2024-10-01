// Project UID 1d9f47bfc76643019cfbf037641defe1

#include <cassert>
#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <array>
#include <vector>
#include <sstream>
#include <algorithm>
#include "Card.h"
// add any necessary #include or using directives here

// rank and suit names -- do not remove these
constexpr const char* const Card::RANK_TWO;
constexpr const char* const Card::RANK_THREE;
constexpr const char* const Card::RANK_FOUR;
constexpr const char* const Card::RANK_FIVE;
constexpr const char* const Card::RANK_SIX;
constexpr const char* const Card::RANK_SEVEN;
constexpr const char* const Card::RANK_EIGHT;
constexpr const char* const Card::RANK_NINE;
constexpr const char* const Card::RANK_TEN;
constexpr const char* const Card::RANK_JACK;
constexpr const char* const Card::RANK_QUEEN;
constexpr const char* const Card::RANK_KING;
constexpr const char* const Card::RANK_ACE;

constexpr const char* const Card::SUIT_SPADES;
constexpr const char* const Card::SUIT_HEARTS;
constexpr const char* const Card::SUIT_CLUBS;
constexpr const char* const Card::SUIT_DIAMONDS;

// add your code below


// NOTE: We HIGHLY recommend you check out the operator overloading
// tutorial in the project spec (see the appendices) before implementing
// the following operator overload functions:
//   operator<<
//   operator<
//   operator<=
//   operator>
//   operator>=
//   operator==
//   operator!=

Card::Card() {
    rank = "Two";
    suit = "Spades";
}

Card::Card(const std::string &rank_in, const std::string &suit_in) {
    rank = rank_in;
    suit = suit_in;
}

std::string Card::get_rank() const {
    return rank;
}

std::string Card::get_suit() const {
    return suit;
}

std::string Card::get_suit(const std::string &trump) const {
    if (is_left_bower(trump)) {
            return trump;
        }
    return suit;
}

bool Card::is_face() const {
    if (rank == "Jack" || rank == "Queen" || rank == "King" || rank == "Ace") {
        return true;
    }
    return false;
}

bool Card::is_right_bower(const std::string &trump) const {
    if (rank == "Jack" && suit == trump) {
        return true;
    }
    return false;
}

bool Card::is_left_bower(const std::string &trump) const{
    if (rank == "Jack" && suit == Suit_next(trump)) {
        return true;
    }
    return false;
}

bool Card::is_trump(const std::string &trump) const {
    if (suit == trump || is_left_bower(trump)) {
        return true;
    }
    return false;
}

bool operator<(const Card &lhs, const Card &rhs) {
    std::array<std::string, 4> suit_value =
    {"Spades", "Hearts", "Clubs", "Diamonds"};
    std::array<std::string, 13> rank_value = {"Two", "Three", "Four", "Five",
    "Six", "Seven", "Eight", "Nine", "Ten", "Jack", "Queen", "King", "Ace"};
    if (lhs.get_rank() != rhs.get_rank()) {
        int lhs_rank_value = 0;
        int rhs_rank_value = 0;
        for (int i = 0; i < 13; ++i) {
            if (rank_value[i] == lhs.get_rank()) {
                lhs_rank_value = i;
            }
            if (rank_value[i] == rhs.get_rank()) {
                rhs_rank_value = i;
            }
        }
        if (lhs_rank_value < rhs_rank_value) {
            return true;
        }
        return false;
    }
    else {
        int lhs_suit_value = 0;
        int rhs_suit_value = 0;
        for (int j = 0; j < 4; ++j) {
            if (suit_value[j] == lhs.get_suit()) {
                lhs_suit_value = j;
            }
            if (suit_value[j] == rhs.get_suit()) {
                rhs_suit_value = j;
            }
        }
        if (lhs_suit_value < rhs_suit_value) {
            return true;
        }
    }
    return false;
}

bool operator<=(const Card &lhs, const Card &rhs) {
    if (lhs < rhs || lhs == rhs) {
        return true;
    }
    return false;
}

bool operator>(const Card &lhs, const Card &rhs) {
    if (lhs < rhs || lhs == rhs) {
        return false;
    }
    return true;
}

bool operator>=(const Card &lhs, const Card &rhs) {
    if (lhs < rhs) {
        return false;
    }
    return true;
}
bool operator==(const Card &lhs, const Card &rhs) {
    if (lhs.get_rank() == rhs.get_rank() && lhs.get_suit() == rhs.get_suit()) {
        return true;
    }
    return false;
}

bool operator!=(const Card &lhs, const Card &rhs) {
    if (lhs == rhs) {
        return false;
    }
    return true;
}


std::string Suit_next(const std::string &suit) {
    if (suit == "Spades") {
        return "Clubs";
    }
    if (suit == "Clubs") {
        return "Spades";
    }
    if (suit == "Diamonds") {
        return "Hearts";
    }
    return "Diamonds";
}

std::ostream & operator<<(std::ostream &os, const Card &card) {
    os << card.get_rank() << " of " << card.get_suit();
    return os;
}

bool Card_less(const Card &a, const Card &b, const std::string &trump) {
    if (a.is_trump(trump) && b.is_trump(trump)) {
        if (a.is_right_bower(trump)) {
            return false;
        }
        else if (b.is_right_bower(trump)) {
            return true;
        }
        else if (a.is_left_bower(trump)) {
            return false;
        }
        else if (b.is_left_bower(trump)) {
            return true;
        }
        else if (a < b) {
            return true;
        }
        else {
            return false;
        }
    }
    else if (!b.is_trump(trump) && a.is_trump(trump)) {
        return false;
    }
    else if (b.is_trump(trump) && !a.is_trump(trump)) {
        return true;
    }
    else if (a < b) {
        return true;
    }
    else {
        return false;
    }
}

bool Card_less(const Card &a, const Card &b, const Card &led_card,
               const std::string &trump){
    if (a.is_trump(trump) && b.is_trump(trump)) {
        if (a.is_right_bower(trump)) {
            return false;
        }
        if (b.is_right_bower(trump)) {
            return true;
        }
        if (a.is_left_bower(trump)) {
            return false;
        }
        if (b.is_left_bower(trump)) {
            return true;
        }
        if (a < b) {
            return true;
        }
        return false;
    }
    if (a.is_trump(trump) && !b.is_trump(trump)) {
        return false;
    }
    if (!a.is_trump(trump) && b.is_trump(trump)) {
        return true;
    }
    if (a.get_suit(trump) == led_card.get_suit(trump)) {
        if (b.get_suit(trump) == led_card.get_suit(trump)) {
            if (a < b) {
                return true;
            }
            return false;
        }
        return false;
    }
    else if (b.get_suit() == led_card.get_suit()) {
        return true;
    }
    if (a < b) {
        return true;
    }
    return false;
}
