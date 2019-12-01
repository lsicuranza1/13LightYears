package game.patterns.state;

public class StatoInEsecuzione implements Stato {
 
    @Override
	public void gestioneStato(Modalita modalita, String stato) {
		// TODO Auto-generated method stub
		if (stato.equals("pausa"))
			modalita.setStatoModalita(new StatoPausa());
		else if(stato.equals("game_over"))
			modalita.setStatoModalita(new StatoGameOver());
	}
 
}
