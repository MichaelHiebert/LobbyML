# LobbyML

A repo to track ML research for LobbyView.

## Updates

### 29 May

The current repo consists of `model.py` and data `data/data.npy`. `data` is a numpy array of shape `20_000 x 127`, where each row corresponds to one issue / for-against relation.

The 127-length vector *t* can be broken down as follows:

1. t[0] == label (0 or 1) for for/against
2. t[1:69] == a one hot containing the issueOrdi of this issue
3. t[69] == the number of lobbyists lobbying on this bill
4. t[70:88] == lobby vector comprised of the following:
    - `xPca, yPca, zPca, nYearsNormalized, nScodesNormalized, nYears, nIssues, sumAmount, nLegislatorWorkedForCommittees, nLegislatorWorkedForCommittees, nDistinctBillCosponsor, nDistinctClients, nDistinctRegistrants, percentClientLocalOrStateGov, percentInternalVsExternalLobbying, nGovEntititesLobbied, nRevolvingDoorLobbyistInTeam, nLegislatorsForTripleLobbyists
5. t[88] == the date difference (in days) between report submission and bill introduction
6. t[89] == the number of terms the bill's sponsor has been in office
7. t[90:92] == a one hot containing the party affiliation of the sponsor at the time of bill introduction
8. t[92:] == a one-hot containing the committee information of this bill

As of today, it is not normalized.
