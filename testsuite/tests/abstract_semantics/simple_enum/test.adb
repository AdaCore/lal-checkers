procedure Test is
   type My_State is (Init, Do_Stuff_1, Do_Stuff_2, Finalize);
   state : My_State := Init;
   new_state : My_State;
begin
   while state /= Finalize loop
      if state = Init then
         new_state := Do_Stuff_1;
      end if;
      if state = Do_Stuff_1 then
         new_state := Finalize;
      end if;

      state := new_state;
   end loop;
end Ex1;
