package game.patterns.state;

public class StatoStart implements Stato {
 
    @Override
	public void gestioneStato(Modalita modalita, String stato) {
		// TODO Auto-generated method stub
		if (stato.equals("in_esecuzione"))
			modalita.setStatoModalita(new StatoInEsecuzione());
		else if (stato.equals("game_over"))
			modalita.setStatoModalita(new StatoGameOver());
	}
 
}
